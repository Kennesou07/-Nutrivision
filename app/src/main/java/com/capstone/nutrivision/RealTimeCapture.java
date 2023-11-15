package com.capstone.nutrivision;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Camera;
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
    private Mat mRgba,mGray;
    private CameraBridgeViewBase mOpenCvCameraView;
    private objectDetectorClass objectDetectorClass;
    private List<String> labelList;  // Declare labelList as a class variable
    private AssetManager assetManager;  // Declare assetManager as a class variable
    private String labelpath = "label.txt";  // Declare labelpath and initialize it with the correct file name

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface.SUCCESS:{
                    Log.i(TAG,"OpenCv is Loaded");
                    mOpenCvCameraView.enableView();
                }
                default:{
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };
    public RealTimeCapture(){
        Log.i(TAG,"Instantiated new " + this.getClass());
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
        try{
            //input size for 640 for this model
            objectDetectorClass = new objectDetectorClass(getAssets(),"yolov4-tiny-416-fp16.tflite",labelpath,416);
            labelList = loadLabelList(assetManager, labelpath);
            Log.d("Recognition", "Label List: " + Arrays.toString(labelList.toArray()));
            Log.d("MainActivity","Model is successfully loaded");
        }
        catch(IOException e){
            e.printStackTrace();
        }
        mOpenCvCameraView.enableFpsMeter();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if(OpenCVLoader.initDebug()){
            Log.d(TAG,"OpenCv Initialization success");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            Log.d(TAG,"OpenCv Initialization failed... try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this, mLoaderCallback);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    public void onCameraViewStarted(int width, int height){
        mRgba = new Mat(height,width, CvType.CV_8UC4);
        mGray = new Mat(height,width,CvType.CV_8UC1);
    }

    public void onCameraViewStopped(){
        mRgba.release();
    }
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        Mat detectedFrame = objectDetectorClass.recognizeImage(mRgba);

        return detectedFrame;
    }
}