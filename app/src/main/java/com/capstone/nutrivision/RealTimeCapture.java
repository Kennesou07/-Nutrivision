package com.capstone.nutrivision;

import android.app.Activity;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RealTimeCapture extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "RealTimeCapture";
    private static final String MODEL_PATH = "custom_yolov4-416-fp16.tflite";
    private static final int INPUT_SIZE = 416;
    private String labelpath = "9label.txt";

    private Mat mRgba, mGray;
    private CameraBridgeViewBase mOpenCvCameraView;
    private objectDetectorClass objectDetectorClass;
    private List<String> labelList;
    private AssetManager assetManager;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCv is Loaded");
                    mOpenCvCameraView.enableView();
                }
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public RealTimeCapture() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelpath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader br = new BufferedReader(new InputStreamReader(assetManager.open(labelpath)));
        String line;
        while ((line = br.readLine()) != null) {
            labelList.add(line.trim());
        }
        br.close();
        return labelList;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.camera_view);
        mOpenCvCameraView = findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        assetManager = getAssets();
        try {
            objectDetectorClass = new objectDetectorClass(getAssets(), MODEL_PATH, labelpath, INPUT_SIZE);
            labelList = loadLabelList(assetManager, labelpath);
            Log.d(TAG, "Label List: " + Arrays.toString(labelList.toArray()));
            Log.d(TAG, "Model is successfully loaded");
        } catch (IOException e) {
            e.printStackTrace();
        }
        mOpenCvCameraView.enableFpsMeter();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCv Initialization success");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.d(TAG, "OpenCv Initialization failed... try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        Mat detectedFrame = objectDetectorClass.recognizeImage(mRgba);
        return detectedFrame;
    }
}
