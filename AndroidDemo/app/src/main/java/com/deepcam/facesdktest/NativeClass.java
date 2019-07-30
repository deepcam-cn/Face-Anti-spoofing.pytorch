package com.deepcam.facesdktest;

import android.graphics.Bitmap;

/**
 * Created by rushuai on 18-9-5.
 */

public class NativeClass {
    static {
        try {
            System.loadLibrary("native-lib");
        } catch (Exception e) {

        }
    }

    static native int native_init();
    static native int native_test(Bitmap bitmap, byte[] data, int angle);
}
