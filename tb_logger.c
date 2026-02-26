/* ============================================================================
 * tb_logger.c — TensorBoard-Compatible Event Logger
 * ============================================================================
 * Writes TFRecord/TFEvent files that TensorBoard can read directly.
 *
 * File format (TFRecord):
 *   uint64  data_length
 *   uint32  masked_crc32(data_length)
 *   byte    data[data_length]        // serialised tf.Event protobuf
 *   uint32  masked_crc32(data)
 *
 * We hand-roll minimal protobuf encoding so there is zero dependency on
 * protobuf-c or any external library.
 *
 * The assembly math kernels in training_ops.asm handle the hot-path
 * numerical work (norms, thresholds, etc.).  This file handles the I/O-
 * heavy, format-heavy logic that would be painful in pure assembly.
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

/* ---- CRC32C (Castagnoli) ------------------------------------------------- */

static uint32_t crc32c_table[256];
static int      crc32c_table_ready = 0;

static void crc32c_init_table(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (crc & 1 ? 0x82F63B78u : 0);
        crc32c_table[i] = crc;
    }
    crc32c_table_ready = 1;
}

static uint32_t crc32c(const void *buf, size_t len) {
    if (!crc32c_table_ready) crc32c_init_table();
    const uint8_t *p = (const uint8_t *)buf;
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++)
        crc = crc32c_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFF;
}

/* TFRecord uses "masked CRC": rotate then add a constant */
static uint32_t masked_crc32c(const void *buf, size_t len) {
    uint32_t crc = crc32c(buf, len);
    return ((crc >> 15) | (crc << 17)) + 0xA282EAD8u;
}

/* ---- Minimal protobuf encoder -------------------------------------------- */

/* Encode a varint into buf, return bytes written */
static int pb_encode_varint(uint8_t *buf, uint64_t value) {
    int n = 0;
    while (value > 0x7F) {
        buf[n++] = (uint8_t)(value & 0x7F) | 0x80;
        value >>= 7;
    }
    buf[n++] = (uint8_t)value;
    return n;
}

/* field tag: (field_number << 3) | wire_type */
static int pb_encode_tag(uint8_t *buf, uint32_t field, uint32_t wire) {
    return pb_encode_varint(buf, ((uint64_t)field << 3) | wire);
}

/* Encode a double as fixed64 (wire type 1) */
static int pb_encode_double(uint8_t *buf, uint32_t field, double val) {
    int n = pb_encode_tag(buf, field, 1); /* wire type 1 = 64-bit */
    memcpy(buf + n, &val, 8);
    return n + 8;
}

/* Encode a float as fixed32 (wire type 5) */
static int pb_encode_float(uint8_t *buf, uint32_t field, float val) {
    int n = pb_encode_tag(buf, field, 5); /* wire type 5 = 32-bit */
    memcpy(buf + n, &val, 4);
    return n + 4;
}

/* Encode an int64 as varint (wire type 0) */
static int pb_encode_int64(uint8_t *buf, uint32_t field, int64_t val) {
    int n = pb_encode_tag(buf, field, 0);
    n += pb_encode_varint(buf + n, (uint64_t)val);
    return n;
}

/* Encode a length-delimited field (wire type 2).
 * Writes tag + length; caller must append the payload right after. */
static int pb_encode_len_prefix(uint8_t *buf, uint32_t field, uint64_t payload_len) {
    int n = pb_encode_tag(buf, field, 2);
    n += pb_encode_varint(buf + n, payload_len);
    return n;
}

/* ---- Build a tf.Summary protobuf for a single scalar --------------------- */
/*
 * tf.Summary {
 *   repeated Value value = 1;
 * }
 * tf.Summary.Value {
 *   string tag = 1;
 *   float  simple_value = 2;  // for scalars
 * }
 */
static int build_summary(uint8_t *buf, const char *tag, float value) {
    uint8_t inner[512];
    int ilen = 0;

    /* Value.tag (field 1, length-delimited string) */
    size_t tag_len = strlen(tag);
    ilen += pb_encode_len_prefix(inner + ilen, 1, tag_len);
    memcpy(inner + ilen, tag, tag_len);
    ilen += (int)tag_len;

    /* Value.simple_value (field 2, float/fixed32) */
    ilen += pb_encode_float(inner + ilen, 2, value);

    /* Summary.value (field 1, length-delimited submessage) */
    int n = 0;
    n += pb_encode_len_prefix(buf + n, 1, (uint64_t)ilen);
    memcpy(buf + n, inner, ilen);
    n += ilen;

    return n;
}

/* ---- Build a tf.Event protobuf ------------------------------------------- */
/*
 * tf.Event {
 *   double wall_time = 1;
 *   int64  step      = 2;
 *   oneof what {
 *     Summary summary = 5;
 *   }
 * }
 */
static int build_event(uint8_t *buf, double wall_time, int64_t step,
                       const uint8_t *summary, int summary_len) {
    int n = 0;
    n += pb_encode_double(buf + n, 1, wall_time);
    n += pb_encode_int64 (buf + n, 2, step);
    n += pb_encode_len_prefix(buf + n, 5, (uint64_t)summary_len);
    memcpy(buf + n, summary, summary_len);
    n += summary_len;
    return n;
}

/* ---- Write one TFRecord -------------------------------------------------- */
static int write_tfrecord(FILE *fp, const uint8_t *data, size_t data_len) {
    uint64_t len = (uint64_t)data_len;
    uint32_t len_crc  = masked_crc32c(&len, sizeof(len));
    uint32_t data_crc = masked_crc32c(data, data_len);

    if (fwrite(&len,      sizeof(len),      1, fp) != 1) return -1;
    if (fwrite(&len_crc,  sizeof(len_crc),  1, fp) != 1) return -1;
    if (fwrite(data,      data_len,         1, fp) != 1) return -1;
    if (fwrite(&data_crc, sizeof(data_crc), 1, fp) != 1) return -1;
    return 0;
}

/* ---- Public API ---------------------------------------------------------- */

/* Opaque writer handle */
typedef struct {
    FILE *fp;
    char  logdir[512];
    char  filepath[1024];
} TBWriter;

#define MAX_WRITERS 16
static TBWriter writers[MAX_WRITERS];
static int      writers_used = 0;

static double get_wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

/*
 * tb_create_writer — open a new TensorBoard event file.
 * Returns a writer handle (index), or -1 on error.
 */
int tb_create_writer(const char *logdir) {
    if (!logdir || writers_used >= MAX_WRITERS) return -1;

    TBWriter *w = &writers[writers_used];

    /* Copy logdir */
    snprintf(w->logdir, sizeof(w->logdir), "%s", logdir);

    /* Build filepath: <logdir>/events.out.tfevents.<timestamp>.<hostname> */
    char hostname[64] = "localhost";
    /* gethostname can be used but we keep it simple */
    FILE *hf = fopen("/etc/hostname", "r");
    if (hf) {
        if (fgets(hostname, sizeof(hostname), hf)) {
            /* strip newline */
            char *nl = strchr(hostname, '\n');
            if (nl) *nl = '\0';
        }
        fclose(hf);
    }

    time_t now = time(NULL);
    snprintf(w->filepath, sizeof(w->filepath),
             "%s/events.out.tfevents.%ld.%s",
             logdir, (long)now, hostname);

    /* Create directory (system call, best-effort) */
    char cmd[600];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", logdir);
    (void)system(cmd);

    w->fp = fopen(w->filepath, "wb");
    if (!w->fp) return -1;

    /* Write file_version event (required by TensorBoard) */
    uint8_t ev[256];
    int n = 0;
    /* Event.wall_time (field 1) */
    n += pb_encode_double(ev + n, 1, get_wall_time());
    /* Event.file_version (field 3, string) — "brain.Event:2" */
    const char *ver = "brain.Event:2";
    size_t ver_len = strlen(ver);
    n += pb_encode_len_prefix(ev + n, 3, ver_len);
    memcpy(ev + n, ver, ver_len);
    n += (int)ver_len;

    write_tfrecord(w->fp, ev, (size_t)n);
    fflush(w->fp);

    int handle = writers_used;
    writers_used++;
    return handle;
}

/*
 * tb_add_scalar — log a scalar value.
 */
int tb_add_scalar(int handle, const char *tag, float value, int64_t step) {
    if (handle < 0 || handle >= writers_used) return -1;
    TBWriter *w = &writers[handle];
    if (!w->fp) return -1;

    uint8_t summary[1024];
    int slen = build_summary(summary, tag, value);

    uint8_t event[2048];
    int elen = build_event(event, get_wall_time(), step, summary, slen);

    int rc = write_tfrecord(w->fp, event, (size_t)elen);
    fflush(w->fp);
    return rc;
}

/*
 * tb_add_scalars — log multiple scalars under a main tag.
 * tag_prefix: e.g. "loss"
 * subtags:    e.g. ["train", "val"]
 * values:     corresponding float values
 * count:      number of scalars
 */
int tb_add_scalars(int handle, const char *tag_prefix,
                   const char **subtags, const float *values,
                   int count, int64_t step) {
    for (int i = 0; i < count; i++) {
        char full_tag[256];
        snprintf(full_tag, sizeof(full_tag), "%s/%s", tag_prefix, subtags[i]);
        int rc = tb_add_scalar(handle, full_tag, values[i], step);
        if (rc != 0) return rc;
    }
    return 0;
}

/*
 * tb_add_histogram — log a histogram (simplified: just min/max/mean/count).
 * TensorBoard histograms use a specific protobuf but we write scalars as
 * a pragmatic fallback that shows up in the SCALARS tab.
 */
int tb_add_histogram_stats(int handle, const char *tag,
                           const float *data, int count, int64_t step) {
    if (handle < 0 || handle >= writers_used || !data || count <= 0)
        return -1;

    float mn = data[0], mx = data[0], sum = 0.0f;
    for (int i = 0; i < count; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sum += data[i];
    }
    float mean = sum / (float)count;

    char buf[256];
    snprintf(buf, sizeof(buf), "%s/min", tag);
    tb_add_scalar(handle, buf, mn, step);
    snprintf(buf, sizeof(buf), "%s/max", tag);
    tb_add_scalar(handle, buf, mx, step);
    snprintf(buf, sizeof(buf), "%s/mean", tag);
    tb_add_scalar(handle, buf, mean, step);

    return 0;
}

/*
 * tb_flush — flush pending writes.
 */
int tb_flush(int handle) {
    if (handle < 0 || handle >= writers_used) return -1;
    if (writers[handle].fp) fflush(writers[handle].fp);
    return 0;
}

/*
 * tb_close — close the writer.
 */
int tb_close(int handle) {
    if (handle < 0 || handle >= writers_used) return -1;
    if (writers[handle].fp) {
        fclose(writers[handle].fp);
        writers[handle].fp = NULL;
    }
    return 0;
}

/*
 * tb_get_logdir — return the log directory for a writer.
 */
const char *tb_get_logdir(int handle) {
    if (handle < 0 || handle >= writers_used) return NULL;
    return writers[handle].logdir;
}

/*
 * tb_get_filepath — return the event file path for a writer.
 */
const char *tb_get_filepath(int handle) {
    if (handle < 0 || handle >= writers_used) return NULL;
    return writers[handle].filepath;
}
