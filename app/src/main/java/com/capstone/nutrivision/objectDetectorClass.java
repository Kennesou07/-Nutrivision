package com.capstone.nutrivision;


import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class objectDetectorClass {
    private Interpreter interpreter;
    private List<String> labelList;
    private int INPUT_SIZE;
    private int PIXEL_SIZE = 3;
    private int IMAGE_MEAN = 0;
    private float IMAGE_STD = 255.0f;
    private GpuDelegate gpuDelegate;
    private int height = 0;
    private int width = 0;
    objectDetectorClass(AssetManager assetManager, String modelPath, String labelpath, int inputSize) throws IOException{
        INPUT_SIZE = inputSize;
        Interpreter.Options options = new Interpreter.Options();
//        gpuDelegate = new GpuDelegate();
//        options.addDelegate(gpuDelegate);
        try {
            gpuDelegate = new GpuDelegate();
//            options.addDelegate(gpuDelegate);
            Log.d("GPUDelegate", "GPU delegate initialized successfully");
        } catch (Exception e) {
            Log.e("GPUDelegateError", "Error initializing GPU delegate", e);
        }
        options.setNumThreads(4);
        interpreter = new Interpreter( loadModelFile(assetManager,modelPath),options);
        labelList = loadLabelList(assetManager,labelpath);
    }
    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException{
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader((assetManager.open(labelPath))));
        String line;
        while((line = reader.readLine()) != null){
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }
    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException{
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
    public Mat recognizeImage(Mat mat_image){
        try{
            Log.d("Recognition", "Start recognition");
            Mat rotated_mat_image = new Mat();
            Mat a = mat_image.t();
            Core.flip(a,rotated_mat_image,1);
            a.release();
            // Core.flip(mat_image.t(),rotated_mat_image,1);
            Bitmap bitmap = null;
            bitmap = Bitmap.createBitmap(rotated_mat_image.cols(),rotated_mat_image.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rotated_mat_image, bitmap);
            height = bitmap.getHeight();
            width = bitmap.getWidth();
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);
            Object[] input = new Object[1];
            input[0] = byteBuffer;
            Map<Integer,Object> output_map = new TreeMap<>();

            // Print the number of output tensors
            int outputTensorCount = interpreter.getOutputTensorCount();
            Log.d("Tensor Shapes", "Number of output tensors: " + outputTensorCount);

            // Print the indices of output tensors
            for (int i = 0; i < outputTensorCount; i++) {
                Log.d("Tensor Shapes", "Output Tensor at index " + i + ": " + interpreter.getOutputTensor(i).name());
            }
            int boxesIndex = 0;
            int scoresIndex = 1;

            int[] boxesShape = interpreter.getOutputTensor(boxesIndex).shape();
            int[] scoresShape = interpreter.getOutputTensor(scoresIndex).shape();

            float[][][] boxes = new float[boxesShape[0]][boxesShape[1]][boxesShape[2]];
            float[][][] scores = new float[scoresShape[0]][scoresShape[1]][scoresShape[2]];


            output_map.put(boxesIndex, boxes);
            output_map.put(scoresIndex, scores);

            Object value = output_map.get(0); // Use index 0 for the boxes tensor
            Object Object_class = output_map.get(1); // Use index 1 for the classes tensor
            Log.d("Recognition", "Before processing outputs");

            if (value != null) {
                Log.d("Object Value", value.getClass().getName());
                Log.d("Object Dimensions - Boxes", Arrays.toString(getDimensions(value)));
            } else {
                Log.e("Recognition", "Value object is null.");
            }

            if (Object_class != null) {
                Log.d("Object Dimensions - Classes", Arrays.toString(getDimensions(Object_class)));
            } else {
                Log.e("Recognition", "Object_class is null.");
            }

            if (value instanceof float[][][]) {
                float[][][] boxesArray = (float[][][]) value;

                if (Object_class instanceof float[][][]) {
                    float[][][] Object_classArray = (float[][][]) Object_class;

                    if (scores instanceof float[][][]) {
                        float[][][] scoresArray = (float[][][]) scores;
                        Log.d("Recognition", "Boxes Array: " + Arrays.deepToString(boxesArray));
                        Log.d("Recognition", "Object Class Array: " + Arrays.deepToString(Object_classArray));
                        Log.d("Recognition", "Scores Array: " + Arrays.deepToString(scoresArray));
                        for (int i = 0; i < getDimensions(Object_class)[2]; i++) {
                            float class_value = Object_classArray[0][0][i];
                            float confidence_score = scoresArray[0][0][i];
                            Log.d("Detection Result", "Class: " + class_value + ", Confidence: " + confidence_score +
                                    ", Box: " + Arrays.toString(boxesArray[0][i]));
                            if (confidence_score > 0.1) {
                                float[] box1 = boxesArray[0][i];

                                float top = box1[0] * height;
                                float left = box1[1] * width;
                                float bottom = box1[2] * height;
                                float right = box1[3] * width;
                                Log.d("Recognition", "Detected Object: " + labelList.get((int) class_value));
                                Imgproc.rectangle(rotated_mat_image, new Point(left, top), new Point(right, bottom), new Scalar(0, 255, 0, 255), 2);
                                Imgproc.putText(rotated_mat_image, labelList.get((int) class_value), new Point(left, top), 3, 1, new Scalar(0, 255, 0, 255), 2);
                            }
                        }
                    } else {
                        Log.e("Recognition", "Invalid format for the scores tensor.");
                    }
                } else {
                    Log.e("Recognition", "Invalid format for the object number tensor.");
                }
            } else {
                Log.e("Recognition", "Invalid format for the boxes tensor.");
            }

            Log.d("Recognition", "After processing outputs");

            Mat b = rotated_mat_image.t();
            Core.flip(b,mat_image,0);
            b.release();
//        Core.flip(rotated_mat_image.t(),mat_image,0);
            Log.d("Recognition","End recognition");
        }
        catch(Exception e){
            Log.e("Recognition","Exception During Recognition", e);
        }
        return mat_image;
    }
    private static int[] getDimensions(Object array) {
        int dimensions = 0;
        Class<?> type = array.getClass();
        while (type.isArray()) {
            dimensions++;
            type = type.getComponentType();
        }
        int[] result = new int[dimensions];
        for (int i = 0; i < dimensions; i++) {
            result[i] = Array.getLength(array);
            array = Array.get(array, 0);
        }
        return result;
    }

    public Mat recognizePhoto(Mat mat_image){
        Bitmap bitmap = null;
        bitmap = Bitmap.createBitmap(mat_image.cols(),mat_image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat_image, bitmap);
        height = bitmap.getHeight();
        width = bitmap.getWidth();
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);
        Object[] input = new Object[1];
        input[0] = byteBuffer;
        Map<Integer,Object> output_map = new TreeMap<>();
        // Print the number of output tensors
        int outputTensorCount = interpreter.getOutputTensorCount();
        Log.d("Tensor Shapes", "Number of output tensors: " + outputTensorCount);

        // Print the indices of output tensors
        for (int i = 0; i < outputTensorCount; i++) {
            Log.d("Tensor Shapes", "Output Tensor at index " + i + ": " + interpreter.getOutputTensor(i).name());
        }
        int boxesIndex = 0;
        int scoresIndex = 1;

        int[] boxesShape = interpreter.getOutputTensor(boxesIndex).shape();
        int[] scoresShape = interpreter.getOutputTensor(scoresIndex).shape();

        float[][][] boxes = new float[boxesShape[0]][boxesShape[1]][boxesShape[2]];
        float[][][] scores = new float[scoresShape[0]][scoresShape[1]][scoresShape[2]];


        output_map.put(boxesIndex, boxes);
        output_map.put(scoresIndex, scores);

        Object value = output_map.get(0); // Use index 0 for the boxes tensor
        Object Object_class = output_map.get(1); // Use index 1 for the classes tensor
        Log.d("Recognition", "Before processing outputs");

        if (value != null) {
            Log.d("Object Value", value.getClass().getName());
            Log.d("Object Dimensions - Boxes", Arrays.toString(getDimensions(value)));
        } else {
            Log.e("Recognition", "Value object is null.");
        }

        if (Object_class != null) {
            Log.d("Object Dimensions - Classes", Arrays.toString(getDimensions(Object_class)));
        } else {
            Log.e("Recognition", "Object_class is null.");
        }

        if (value instanceof float[][][]) {
            float[][][] boxesArray = (float[][][]) value;

            if (Object_class instanceof float[][][]) {
                float[][][] Object_classArray = (float[][][]) Object_class;

                if (scores instanceof float[][][]) {
                    float[][][] scoresArray = (float[][][]) scores;

                    for (int i = 0; i < getDimensions(Object_class)[2]; i++) {
                        float class_value = Object_classArray[0][0][i];
                        float confidence_score = scoresArray[0][0][i];

                        if (confidence_score > 0.5) {
                            float[] box1 = boxesArray[0][i];

                            float top = box1[0] * height;
                            float left = box1[1] * width;
                            float bottom = box1[2] * height;
                            float right = box1[3] * width;
                            Imgproc.rectangle(mat_image, new Point(left, top), new Point(right, bottom), new Scalar(0, 255, 0, 255), 2);
                            Imgproc.putText(mat_image, labelList.get((int) class_value), new Point(left, top), 3, 1, new Scalar(0, 255, 0, 255), 2);
                        }
                    }
                } else {
                    Log.e("Recognition", "Invalid format for the scores tensor.");
                }
            } else {
                Log.e("Recognition", "Invalid format for the object number tensor.");
            }
        } else {
            Log.e("Recognition", "Invalid format for the boxes tensor.");
        }

        Log.d("Recognition", "After processing outputs");

        return mat_image;
    }
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap){
        ByteBuffer byteBuffer;
        int quant = 1;
        int size_images = INPUT_SIZE;
        if(quant == 0){
            byteBuffer = ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else{
            byteBuffer = ByteBuffer.allocateDirect(4*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel = 0;
        for(int i=0; i<size_images; ++i){
            for(int j=0; j<size_images; ++j){
                final int val = intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte)((val>>16) & 0xFF));
                    byteBuffer.put((byte)((val>>8) & 0xFF));
                    byteBuffer.put((byte)((val) & 0xFF));
                }
                else{
                    byteBuffer.putFloat(((val>>16) & 0xFF));
                    byteBuffer.putFloat(((val>>8) & 0xFF));
                    byteBuffer.putFloat(((val) & 0xFF));
                }
            }
        }
        return byteBuffer;
    }
}