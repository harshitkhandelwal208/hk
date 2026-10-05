// JNI glue between com.hk.HkModel and the C ABI in include/hk.h.
//
// Build it with `zig build jni -Djdk=/path/to/jdk` (the JDK directory that contains include/jni.h).
// The result, libhkjni, depends on libhk and is loaded by HkModel after it.

#include <jni.h>
#include <stdint.h>
#include <string.h>

#include "../../../include/hk.h"

#define H(x) ((hk_reader_t*)(intptr_t)(x))
#define W(x) ((hk_writer_t*)(intptr_t)(x))
#define JNI(ret, name) JNIEXPORT ret JNICALL Java_com_hk_HkModel_##name

static jobject direct_buffer(JNIEnv* env, const void* ptr, uint64_t size) {
    if (!ptr || size == 0) return NULL;
    return (*env)->NewDirectByteBuffer(env, (void*)ptr, (jlong)size);
}

JNI(jlong, nativeOpen)(JNIEnv* env, jclass cls, jstring path) {
    (void)cls;
    const char* p = (*env)->GetStringUTFChars(env, path, NULL);
    if (!p) return 0;
    hk_reader_t* r = hk_open(p);
    (*env)->ReleaseStringUTFChars(env, path, p);
    return (jlong)(intptr_t)r;
}

JNI(void, nativeClose)(JNIEnv* env, jclass cls, jlong h) {
    (void)env; (void)cls;
    if (h) hk_close(H(h));
}

JNI(jlong, nativeGetTensorCount)(JNIEnv* env, jclass cls, jlong h) {
    (void)env; (void)cls;
    return (jlong)hk_get_tensor_count(H(h));
}

JNI(jobject, nativeGetTensor)(JNIEnv* env, jclass cls, jlong h, jlong index) {
    (void)cls;
    hk_tensor_info_t info;
    if (hk_get_tensor_info(H(h), (uint64_t)index, &info) != 0) return NULL;
    jclass tcls = (*env)->FindClass(env, "com/hk/HkModel$HkTensor");
    if (!tcls) return NULL;
    jmethodID ctor = (*env)->GetMethodID(env, tcls, "<init>", "(JJLjava/lang/String;BBBSF[J)V");
    if (!ctor) return NULL;
    jstring name = (*env)->NewStringUTF(env, info.name ? info.name : "");
    jlongArray shape = (*env)->NewLongArray(env, info.ndim);
    if (!name || !shape) return NULL;
    jlong dims[8];
    for (int i = 0; i < info.ndim && i < 8; i++) dims[i] = (jlong)info.shape[i];
    (*env)->SetLongArrayRegion(env, shape, 0, info.ndim, dims);
    return (*env)->NewObject(env, tcls, ctor, h, index, name, (jbyte)info.storage_type, (jbyte)info.tile_layout,
                             (jbyte)info.sparsity_type, (jshort)info.block_size, (jfloat)info.sparsity_ratio, shape);
}

JNI(jobject, nativeGetTensorData)(JNIEnv* env, jclass cls, jlong h, jlong index) {
    (void)cls;
    uint64_t size = 0;
    const void* p = hk_get_tensor_data(H(h), (uint64_t)index, &size);
    return direct_buffer(env, p, size);
}

JNI(jobject, nativeGetTensorResidual)(JNIEnv* env, jclass cls, jlong h, jlong index) {
    (void)cls;
    uint64_t size = 0;
    const void* p = hk_get_tensor_residual(H(h), (uint64_t)index, &size);
    return direct_buffer(env, p, size);
}

JNI(jobject, nativeGetTensorScales)(JNIEnv* env, jclass cls, jlong h, jlong index) {
    (void)cls;
    uint64_t size = 0;
    const void* p = hk_get_tensor_scales(H(h), (uint64_t)index, &size);
    return direct_buffer(env, p, size);
}

JNI(jobject, nativeGetTensorRawPtr)(JNIEnv* env, jclass cls, jlong h, jlong index) {
    (void)cls;
    uint64_t size = 0;
    const void* p = hk_get_tensor_raw_ptr(H(h), (uint64_t)index, &size);
    return direct_buffer(env, p, size);
}

JNI(jint, nativeDequantizeF32)(JNIEnv* env, jclass cls, jlong h, jlong index, jint with_residual, jobject out, jlong count) {
    (void)cls;
    float* buf = (float*)(*env)->GetDirectBufferAddress(env, out);
    if (!buf) return -1;
    jlong cap = (*env)->GetDirectBufferCapacity(env, out);
    if (cap < count) return -1; // capacity is counted in elements, here floats
    return hk_dequantize_f32(H(h), (uint64_t)index, with_residual, buf, (uint64_t)count);
}

JNI(jstring, nativeGetMetadataString)(JNIEnv* env, jclass cls, jlong h, jstring key) {
    (void)cls;
    const char* k = (*env)->GetStringUTFChars(env, key, NULL);
    if (!k) return NULL;
    const char* v = hk_get_metadata_string(H(h), k);
    (*env)->ReleaseStringUTFChars(env, key, k);
    return v ? (*env)->NewStringUTF(env, v) : NULL;
}

JNI(jint, nativeGetMetadataInt)(JNIEnv* env, jclass cls, jlong h, jstring key, jlongArray out) {
    (void)cls;
    const char* k = (*env)->GetStringUTFChars(env, key, NULL);
    if (!k) return -1;
    int64_t v = 0;
    int r = hk_get_metadata_int(H(h), k, &v);
    (*env)->ReleaseStringUTFChars(env, key, k);
    if (r == 0) {
        jlong jv = (jlong)v;
        (*env)->SetLongArrayRegion(env, out, 0, 1, &jv);
    }
    return r;
}

JNI(jint, nativeGetMetadataFloat)(JNIEnv* env, jclass cls, jlong h, jstring key, jdoubleArray out) {
    (void)cls;
    const char* k = (*env)->GetStringUTFChars(env, key, NULL);
    if (!k) return -1;
    double v = 0;
    int r = hk_get_metadata_float(H(h), k, &v);
    (*env)->ReleaseStringUTFChars(env, key, k);
    if (r == 0) {
        jdouble jv = v;
        (*env)->SetDoubleArrayRegion(env, out, 0, 1, &jv);
    }
    return r;
}

JNI(jint, nativeGetMetadataBool)(JNIEnv* env, jclass cls, jlong h, jstring key, jintArray out) {
    (void)cls;
    const char* k = (*env)->GetStringUTFChars(env, key, NULL);
    if (!k) return -1;
    int v = 0;
    int r = hk_get_metadata_bool(H(h), k, &v);
    (*env)->ReleaseStringUTFChars(env, key, k);
    if (r == 0) {
        jint jv = v;
        (*env)->SetIntArrayRegion(env, out, 0, 1, &jv);
    }
    return r;
}

JNI(jint, nativeIsSharded)(JNIEnv* env, jclass cls, jlong h) { (void)env; (void)cls; return hk_reader_is_sharded(H(h)); }
JNI(jint, nativeGetSplitIndex)(JNIEnv* env, jclass cls, jlong h) { (void)env; (void)cls; return hk_reader_get_split_index(H(h)); }
JNI(jint, nativeGetSplitCount)(JNIEnv* env, jclass cls, jlong h) { (void)env; (void)cls; return hk_reader_get_split_count(H(h)); }
JNI(jint, nativeIsRawStorage)(JNIEnv* env, jclass cls, jlong h) { (void)env; (void)cls; return hk_is_raw_storage(H(h)); }
JNI(jint, nativeIsUniversalPageAligned)(JNIEnv* env, jclass cls, jlong h) { (void)env; (void)cls; return hk_is_universal_page_aligned(H(h)); }
JNI(jint, nativeGetFileAlignment)(JNIEnv* env, jclass cls, jlong h) { (void)env; (void)cls; return (jint)hk_get_file_alignment(H(h)); }

JNI(jint, nativePatchMetadataInPlace)(JNIEnv* env, jclass cls, jstring path, jstring key, jstring val) {
    (void)cls;
    const char* p = (*env)->GetStringUTFChars(env, path, NULL);
    const char* k = (*env)->GetStringUTFChars(env, key, NULL);
    const char* v = (*env)->GetStringUTFChars(env, val, NULL);
    int r = -1;
    if (p && k && v) r = hk_metadata_patch_in_place(p, k, v);
    if (p) (*env)->ReleaseStringUTFChars(env, path, p);
    if (k) (*env)->ReleaseStringUTFChars(env, key, k);
    if (v) (*env)->ReleaseStringUTFChars(env, val, v);
    return r;
}

JNI(jlong, nativeGetAppendixCount)(JNIEnv* env, jclass cls, jlong h) {
    (void)env; (void)cls;
    return (jlong)hk_appendix_get_count(H(h));
}

JNI(jobject, nativeGetAppendixEntry)(JNIEnv* env, jclass cls, jlong h, jlong index) {
    (void)cls;
    hk_appendix_entry_t e;
    if (hk_appendix_get_entry(H(h), (uint64_t)index, &e) != 0) return NULL;
    jclass ecls = (*env)->FindClass(env, "com/hk/HkModel$HkAppendixEntry");
    if (!ecls) return NULL;
    jmethodID ctor = (*env)->GetMethodID(env, ecls, "<init>", "(BBIJ[BFFFFLjava/lang/String;Ljava/lang/String;J)V");
    if (!ctor) return NULL;
    jbyteArray hash = (*env)->NewByteArray(env, 32);
    if (!hash) return NULL;
    (*env)->SetByteArrayRegion(env, hash, 0, 32, (const jbyte*)e.parent_hash);
    jstring name = (*env)->NewStringUTF(env, e.name ? e.name : "");
    jstring target = (*env)->NewStringUTF(env, e.target ? e.target : "");
    if (!name || !target) return NULL;
    return (*env)->NewObject(env, ecls, ctor, (jbyte)e.entry_type, (jbyte)e.flags, (jint)e.generation, (jlong)e.timestamp,
                             hash, (jfloat)e.metric_loss, (jfloat)e.metric_acc, (jfloat)e.metric_pass,
                             (jfloat)e.metric_custom, name, target, (jlong)e.data_size);
}

JNI(jint, nativeRollback)(JNIEnv* env, jclass cls, jstring path, jint gen) {
    (void)cls;
    const char* p = (*env)->GetStringUTFChars(env, path, NULL);
    if (!p) return -1;
    int r = hk_appendix_rollback(p, (uint32_t)gen);
    (*env)->ReleaseStringUTFChars(env, path, p);
    return r;
}

// Writer

JNI(jlong, nativeWriterCreate)(JNIEnv* env, jclass cls, jlong alignment) {
    (void)env; (void)cls;
    return (jlong)(intptr_t)hk_writer_create((uint64_t)alignment);
}

JNI(void, nativeWriterDestroy)(JNIEnv* env, jclass cls, jlong w) {
    (void)env; (void)cls;
    if (w) hk_writer_destroy(W(w));
}

JNI(void, nativeWriterSetSharding)(JNIEnv* env, jclass cls, jlong w, jint idx, jint count) {
    (void)env; (void)cls;
    hk_writer_set_sharding(W(w), (uint16_t)idx, (uint16_t)count);
}

JNI(void, nativeWriterSetRawStorage)(JNIEnv* env, jclass cls, jlong w, jint on) {
    (void)env; (void)cls;
    hk_writer_set_raw_storage(W(w), on);
}

JNI(jint, nativeWriterAddMetadataString)(JNIEnv* env, jclass cls, jlong w, jstring key, jstring val) {
    (void)cls;
    const char* k = (*env)->GetStringUTFChars(env, key, NULL);
    const char* v = (*env)->GetStringUTFChars(env, val, NULL);
    int r = -1;
    if (k && v) r = hk_writer_add_metadata_string(W(w), k, v);
    if (k) (*env)->ReleaseStringUTFChars(env, key, k);
    if (v) (*env)->ReleaseStringUTFChars(env, val, v);
    return r;
}

JNI(jint, nativeWriterAddMetadataInt)(JNIEnv* env, jclass cls, jlong w, jstring key, jlong val) {
    (void)cls;
    const char* k = (*env)->GetStringUTFChars(env, key, NULL);
    if (!k) return -1;
    int r = hk_writer_add_metadata_int(W(w), k, (int64_t)val);
    (*env)->ReleaseStringUTFChars(env, key, k);
    return r;
}

JNI(jint, nativeWriterAddMetadataFloat)(JNIEnv* env, jclass cls, jlong w, jstring key, jdouble val) {
    (void)cls;
    const char* k = (*env)->GetStringUTFChars(env, key, NULL);
    if (!k) return -1;
    int r = hk_writer_add_metadata_float(W(w), k, (double)val);
    (*env)->ReleaseStringUTFChars(env, key, k);
    return r;
}

JNI(jint, nativeWriterAddMetadataBool)(JNIEnv* env, jclass cls, jlong w, jstring key, jint val) {
    (void)cls;
    const char* k = (*env)->GetStringUTFChars(env, key, NULL);
    if (!k) return -1;
    int r = hk_writer_add_metadata_bool(W(w), k, val);
    (*env)->ReleaseStringUTFChars(env, key, k);
    return r;
}

JNI(jint, nativeWriterAddTensor)(JNIEnv* env, jclass cls, jlong w, jstring name, jbyte st, jbyte tile, jbyte sp, jbyte ndim,
                                 jlongArray shape, jobject data, jlong data_len, jfloat ratio) {
    (void)cls;
    const char* n = (*env)->GetStringUTFChars(env, name, NULL);
    if (!n) return -1;
    int r = -1;
    uint64_t dims[8] = {0};
    jlong tmp[8] = {0};
    const uint8_t* buf = data ? (const uint8_t*)(*env)->GetDirectBufferAddress(env, data) : NULL;
    if (ndim >= 0 && ndim <= 8 && (buf || data_len == 0)) {
        (*env)->GetLongArrayRegion(env, shape, 0, ndim, tmp);
        for (int i = 0; i < ndim; i++) dims[i] = (uint64_t)tmp[i];
        r = hk_writer_add_tensor(W(w), n, (uint8_t)st, (uint8_t)tile, (uint8_t)sp, (uint8_t)ndim, dims, buf, (uint64_t)data_len, ratio);
    }
    (*env)->ReleaseStringUTFChars(env, name, n);
    return r;
}

JNI(jint, nativeWriterWriteToFile)(JNIEnv* env, jclass cls, jlong w, jstring path) {
    (void)cls;
    const char* p = (*env)->GetStringUTFChars(env, path, NULL);
    if (!p) return -1;
    int r = hk_writer_write_to_file(W(w), p);
    (*env)->ReleaseStringUTFChars(env, path, p);
    return r;
}
