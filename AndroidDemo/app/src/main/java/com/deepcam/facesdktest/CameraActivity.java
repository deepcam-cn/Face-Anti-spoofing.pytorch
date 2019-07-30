package com.deepcam.facesdktest;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.os.Build;
import android.os.Handler;
import android.os.Message;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CameraActivity extends AppCompatActivity implements View.OnClickListener {
    private static final String TAG = "CameraActivity";
    private static final int PERMISSIONS_REQUEST = 1;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final String PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;

    private ImageView mPreviewRect;
    private ImageView mFaceView;
    private Bitmap mFaceBitmap;
    private int mPreviewWidth;
    private int mPreviewHeight;
    private int mCameraId = 0;
    private SurfaceTexture mSurfaceTexture;
    private Camera mCamera;

    private RenderScript mRs;
    private ScriptIntrinsicYuvToRGB mYuvToRgbIntrinsic;
    private Type.Builder mYuvType, mRgbaType;
    private Allocation mIn, mOut;

    private static final int MSG_UPDATE_FACE_UI = 100;
    private static final int CAMERA_ANGLE = 270;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_camera);
        mPreviewRect = (ImageView)findViewById(R.id.preview);
        mFaceView = (ImageView)findViewById(R.id.face_info);
        if (hasPermission()) {
            openFrontCamera();
        } else {
            requestPermission();
        }
        WindowManager wm = (WindowManager)this.getSystemService(Context.WINDOW_SERVICE);
        ViewGroup.LayoutParams params = mPreviewRect.getLayoutParams();
        params.width = wm.getDefaultDisplay().getWidth();
        if(CAMERA_ANGLE == 270 || CAMERA_ANGLE == 90) {
            params.height = (int) (params.width * (640.0f / 480.0f));
        } else {
            params.height = (int) (params.width * (480.0f / 640.0f));
        }
        Log.d(TAG, "surface view width = " + params.width + ", height = " + params.height);
        mPreviewRect.setLayoutParams(params);
        mFaceView.setLayoutParams(params);
        mFaceView.setVisibility(View.GONE);
        mRs = RenderScript.create(this);
        mYuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(mRs, Element.U8_4(mRs));
        NativeClass.native_init();
    }

    @Override
    protected void onResume() {
        super.onResume();
        startPreview();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(mCamera != null) {
            mCamera.stopPreview();
        }
    }


    @Override
    public void onClick(View arg0) {
        switch (arg0.getId()) {

        }
    }

    private void openFrontCamera() {
        if(mCamera != null) {
            return;
        }
        mCameraId = -1;
        int cameraNum = Camera.getNumberOfCameras();
        for(int i = 0; i < cameraNum; i++) {
            Camera.CameraInfo tmp = new Camera.CameraInfo();
            Camera.getCameraInfo(i, tmp);
            if(tmp.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                mCameraId = i;
                break;
            }
        }
        if(mCameraId < 0) {
            Log.e(TAG, "can not find the front camera");
            //return;
            mCameraId = 0;
        }
        Log.d(TAG, "the front camera id : " + mCameraId);
        mCamera = Camera.open(mCameraId);
    }

    public static Bitmap rotateBitmap(Bitmap bitmap, int degrees) {
        if (degrees == 0 || null == bitmap) {
            return bitmap;
        }
        Matrix matrix = new Matrix();
        matrix.setRotate(degrees, bitmap.getWidth() / 2, bitmap.getHeight() / 2);
        Bitmap bmp = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        if (null != bitmap) {
            bitmap.recycle();
        }
        return bmp;
    }

    private Camera.PreviewCallback dataCallback = new Camera.PreviewCallback() {
        private Paint mPaint = new Paint(Paint.FAKE_BOLD_TEXT_FLAG);
        @Override
        public void onPreviewFrame(byte[] data, Camera camera) {
            if (mYuvType == null)
            {
                mYuvType = new Type.Builder(mRs, Element.U8(mRs)).setX(data.length);
                mIn = Allocation.createTyped(mRs, mYuvType.create(), Allocation.USAGE_SCRIPT);

                mRgbaType = new Type.Builder(mRs, Element.RGBA_8888(mRs)).setX(mPreviewWidth).setY(mPreviewHeight);
                mOut = Allocation.createTyped(mRs, mRgbaType.create(), Allocation.USAGE_SCRIPT);
            }
            mIn.copyFrom(data);

            mYuvToRgbIntrinsic.setInput(mIn);
            mYuvToRgbIntrinsic.forEach(mOut);
            Bitmap bmpout = Bitmap.createBitmap(mPreviewWidth, mPreviewHeight, Bitmap.Config.ARGB_8888);
            mOut.copyTo(bmpout);
            bmpout = rotateBitmap(bmpout, CAMERA_ANGLE);
            int ret = NativeClass.native_test(bmpout, data, CAMERA_ANGLE);
            Canvas mCanvas = new Canvas(bmpout);
            mPaint.setTextSize(25);
            if(ret == 1) {
                mPaint.setColor(Color.BLUE);
                mCanvas.drawText("PLEASE KEEP STATIC", 120, 50, mPaint);
            } else if(ret == 0) {
                mPaint.setColor(Color.BLACK);
                mCanvas.drawText("NO FACE", 200, 50, mPaint);
            } else if(ret == 100) {
                mPaint.setColor(Color.RED);
                mCanvas.drawText("FAKE", 200, 50, mPaint);
            } else if(ret == 101) {
                mPaint.setColor(Color.GREEN);
                mCanvas.drawText("GENUINE", 200, 50, mPaint);
            }
            mPreviewRect.setImageBitmap(bmpout);
        }
    };

    private void startPreview() {
        if(mCamera == null) {
            return;
        }
        Camera.Parameters parameters = mCamera.getParameters();
        List<Camera.Size> previewSizes = parameters.getSupportedPreviewSizes();
        for(Camera.Size size : previewSizes) {
            Log.d(TAG, "support mPreviewWidth = " + size.width + ", mPreviewHeight = " + size.height);
            if(size.width == 640 && size.height == 480) {
                mPreviewWidth = size.width;
                mPreviewHeight = size.height;
                break;
            }
        }
        if(mPreviewWidth == 0 || mPreviewHeight == 0) {
            //Camera.Size tmp = parameters.getPreferredPreviewSizeForVideo();
            mPreviewHeight = 640;
            mPreviewWidth = 480;
        }
        Log.d(TAG, "mPreviewWidth = " + mPreviewWidth + ", mPreviewHeight = " + mPreviewHeight);
        parameters.setPreviewSize(mPreviewWidth, mPreviewHeight);

        List<String> focusModes = parameters.getSupportedFocusModes();
        if(focusModes.contains("continuous-video")){
            parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
        }
        mCamera.setDisplayOrientation(90);
        mCamera.setParameters(parameters);
        try {
            if(mSurfaceTexture == null) {
                mSurfaceTexture = new SurfaceTexture(1000);
            }
            mCamera.setPreviewTexture(mSurfaceTexture);
            mCamera.setPreviewCallback(dataCallback);
            mCamera.startPreview();
        } catch (IOException e) {

        }
    }

    @Override
    public void onRequestPermissionsResult(
            final int requestCode, final String[] permissions, final int[] grantResults) {
        if (requestCode == PERMISSIONS_REQUEST) {
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED
                    && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                openFrontCamera();
            } else {
                requestPermission();
            }
        }
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(PERMISSION_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA) ||
                    shouldShowRequestPermissionRationale(PERMISSION_STORAGE)) {
                Toast.makeText(CameraActivity.this,
                        "Camera AND storage permission are required for this demo", Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[] {PERMISSION_CAMERA, PERMISSION_STORAGE}, PERMISSIONS_REQUEST);
        }
    }
}