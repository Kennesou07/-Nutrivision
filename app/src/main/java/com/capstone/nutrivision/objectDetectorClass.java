package com.capstone.nutrivision;

import static org.tensorflow.lite.DataType.FLOAT32;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
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
    float confidenceThreshold = 0.5f; // Adjust this value as needed

    private GpuDelegate gpuDelegate;
    private int height = 0;
    private int width = 0;

    objectDetectorClass(AssetManager assetManager, String modelPath, String labelpath, int inputSize) throws IOException {
        INPUT_SIZE = inputSize;
        Interpreter.Options options = new Interpreter.Options();
        try {
            gpuDelegate = new GpuDelegate();
            Log.d("GPUDelegate", "GPU delegate initialized successfully");
//            options.addDelegate(gpuDelegate);
        } catch (Exception e) {
            Log.e("GPUDelegateError", "Error initializing GPU delegate", e);
        }
        options.setNumThreads(4);
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);
        labelList = loadLabelList(assetManager, labelpath);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader((assetManager.open(labelPath))));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public Mat recognizeImage(Mat matImage) {
        try {
            Log.d("Recognition", "Start recognition");
            Mat rotatedMatImage = new Mat();
            Mat a = matImage.t();
            Core.flip(a, rotatedMatImage, 1);
            a.release();

            Bitmap bitmap = Bitmap.createBitmap(rotatedMatImage.cols(), rotatedMatImage.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rotatedMatImage, bitmap);
            height = bitmap.getHeight();
            width = bitmap.getWidth();

            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);
            Object[] input = new Object[1];
            input[0] = byteBuffer;

            Map<Integer, Object> outputMap = new HashMap<>();

            int outputTensorCount = interpreter.getOutputTensorCount();
            Log.d("Tensor Shapes", "Number of output tensors: " + outputTensorCount);

            for (int i = 0; i < outputTensorCount; i++) {
                Log.d("Tensor Shapes", "Output Tensor at index " + i + ": " + interpreter.getOutputTensor(i).name());
            }

            if (outputTensorCount == 2) {

                int boxesIndex = 0;
                int scoresIndex = 1;

                int[] boxesShape = interpreter.getOutputTensor(boxesIndex).shape();
                int[] scoresShape = interpreter.getOutputTensor(scoresIndex).shape();

                float[][][] boxes = new float[boxesShape[0]][boxesShape[1]][boxesShape[2]];
                float[][][] scores = new float[scoresShape[0]][scoresShape[1]][scoresShape[2]];

                outputMap.put(boxesIndex, boxes);
                outputMap.put(scoresIndex, scores);

                interpreter.runForMultipleInputsOutputs(input,outputMap);

                Object value = outputMap.get(0);
                Object objectClass = outputMap.get(1);
                Log.d("Recognition", "Before processing outputs");
                if (value != null) {
                    Log.d("Object Value", value.getClass().getName());
                    Log.d("Object Dimensions - Boxes", Arrays.toString(getDimensions(value)));
                } else {
                    Log.e("Recognition", "Value object is null.");
                }

                if (objectClass != null) {
                    Log.d("Object Dimensions - Classes", Arrays.toString(getDimensions(objectClass)));
                } else {
                    Log.e("Recognition", "Object_class is null.");
                }

                if (value instanceof float[][][]) {
                    float[][][] boxesArray = (float[][][]) value;

                    if (objectClass instanceof float[][][]) {
                        float[][][] objectClassArray = (float[][][]) objectClass;

                        if (scores instanceof float[][][]) {
                            float[][][] scoresArray = (float[][][]) scores;
                            Log.d("Recognition", "Boxes Array: " + Arrays.deepToString(boxesArray));
                            Log.d("Recognition", "Object Class Array: " + Arrays.deepToString(objectClassArray));
                            Log.d("Recognition", "Scores Array: " + Arrays.deepToString(scoresArray));
                            // Add these lines after the for loop in the else block
                            Log.d("Debug", "height: " + height);
                            Log.d("Debug", "width: " + width);
                            int numDetections = getDimensions(objectClass)[2];
                            for (int i = 0; i < numDetections; i++) {
                                float classValue = objectClassArray[0][0][i];
                                float[] rawScores = scoresArray[0][0];
                                // Assuming only one confidence score
                                float[] normalizedScores = applySoftmax(rawScores);

                                float confidenceScore = normalizedScores[0];
                                Log.d("Confidence Score", "Confidence: " + confidenceScore);
                                if (confidenceScore > confidenceThreshold) {
                                    float[] box1 = boxesArray[0][i];

                                    float top = box1[0] * height;
                                    float left = box1[1] * width;
                                    float bottom = box1[2] * height;
                                    float right = box1[3] * width;

                                    Log.d("Recognition", "Detected Object: " + labelList.get((int) classValue));
                                    Imgproc.rectangle(rotatedMatImage, new Point(left, top), new Point(right, bottom), new Scalar(0, 255, 0, 255), 2);
                                    Imgproc.putText(rotatedMatImage, labelList.get((int) classValue), new Point(left, top), 3, 1, new Scalar(0, 255, 0, 255), 2);
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

                Mat b = rotatedMatImage.t();
                Core.flip(b, matImage, 0);
                b.release();
            }else{
                int singleTensorIndex = 0;
                DataType outputDataType = interpreter.getOutputTensor(singleTensorIndex).dataType();
                Log.d("DataType: ","DataType: " + outputDataType);
                // Check if the data type is float
                if (outputDataType == FLOAT32) {
                    int[] singleTensorShape = interpreter.getOutputTensor(singleTensorIndex).shape();
                    float[][][] singleTensorData = new float[singleTensorShape[0]][singleTensorShape[1]][singleTensorShape[2]];

                    outputMap.put(singleTensorIndex, singleTensorData);

                    interpreter.runForMultipleInputsOutputs(new Object[]{byteBuffer},outputMap);

                    // Rest of your code remains the same...
                    Object value = outputMap.get(singleTensorIndex);
                    Log.d("Recognition", "Before processing outputs");
                    if (value != null) {
                        Log.d("Object Value", value.getClass().getName());
                        Log.d("Object Dimensions - Boxes", Arrays.toString(getDimensions(value)));
                    } else {
                        Log.e("Recognition", "Value object is null.");
                    }

                    // Now you can process the singleTensorData as before
                    if (singleTensorData!= null && singleTensorData.length > 0) {
                        float classValue = singleTensorData[0][0][0]; // Assuming the class information is at this location
                        float[] rawScores = {singleTensorData[0][0][1]}; // Assuming only one confidence score
                        float[] normalizedScores = applySoftmax(rawScores);

                        float confidenceScore = normalizedScores[0];
                        Log.d("Confidence Score", "Confidence: " + confidenceScore);

                        if (confidenceScore > confidenceThreshold) {
                            float top = singleTensorData[0][0][2] * height;
                            float left = singleTensorData[0][0][3] * width;
                            float bottom = singleTensorData[0][0][4] * height;
                            float right = singleTensorData[0][0][5] * width;

                            Imgproc.rectangle(rotatedMatImage, new Point(left, top), new Point(right, bottom), new Scalar(0, 255, 0, 255), 2);
                            Imgproc.putText(rotatedMatImage, labelList.get((int) classValue) + " " + confidenceScore, new Point(left, top), 3, 1, new Scalar(0, 255, 0, 255), 2);
                            Mat b = rotatedMatImage.t();
                            Core.flip(b, matImage, 0);
                            b.release();
                        }
                    } else {
                        Log.e("Recognition", "Invalid format for single tensor data.");
                    }

                } else {
                    Log.e("Recognition", "Unsupported data type: " + outputDataType);
                }
            }
            Log.d("Recognition", "End recognition");
            rotatedMatImage.release(); // Release the rotatedMatImage Mat after use
            } catch(Exception e){
                Log.e("Recognition", "Exception During Recognition", e);
            }

        return matImage;
    }
    private float[] applySoftmax(float[] scores) {
        int length = scores.length;
        float[] softmaxScores = new float[length];
        float sumExp = 0;

        // Compute the sum of exponentials
        for (int i = 0; i < length; i++) {
            sumExp += Math.exp(scores[i]);
        }

        // Apply softmax
        for (int i = 0; i < length; i++) {
            softmaxScores[i] = (float) (Math.exp(scores[i]) / sumExp);
        }

        return softmaxScores;
    }

    public Mat recognizePhoto(Mat mat_image) {
        // Resize the image using OpenCV functions
        int inputTensorIndex = interpreter.getInputTensor(0).index();
        int[] inputShape = interpreter.getInputTensor(0).shape();
        DataType inputDataType = interpreter.getInputTensor(0).dataType();
        Log.d("Photo Detection","Input Tensor Index: " + inputTensorIndex);
        Log.d("Photo Detection","Input Tensor Shape: " + inputShape);
        Log.d("Photo Detection","Input Data Type: " + inputDataType);
        Imgproc.resize(mat_image, mat_image, new Size(INPUT_SIZE, INPUT_SIZE));

        ByteBuffer byteBuffer = convertMatToByteBuffer(mat_image);
        Object[] input = new Object[]{byteBuffer};

        Map<Integer, Object> output_map = new HashMap<>();
        int boxesIndex = 0;
        int scoresIndex = 1;

        int[] boxesShape = interpreter.getOutputTensor(boxesIndex).shape();
        int[] scoresShape = interpreter.getOutputTensor(scoresIndex).shape();

        float[][][] boxes = new float[1][boxesShape[1]][boxesShape[2]];
        float[][][] scores = new float[1][scoresShape[1]][scoresShape[2]];

        output_map.put(boxesIndex, boxes);
        output_map.put(scoresIndex, scores);

        interpreter.runForMultipleInputsOutputs(input, output_map);

        float[][][] boxesArray = (float[][][]) output_map.get(boxesIndex);
        float[][][] scoresArray = (float[][][]) output_map.get(scoresIndex);

        for (int i = 0; i < scoresArray[0][0].length; i++) {
            float confidence_score = scoresArray[0][0][i];
            Log.d("Detection", "Confidence Score: " + confidence_score);

            // Convert confidence score to percentage
            float confidencePercentage = confidence_score * 100;
            Log.d("Detection", "Confidence Percentage: " + confidencePercentage + "%");

            if (confidencePercentage > confidenceThreshold) {
                float[] box = boxesArray[0][i];

                float top = box[0] * height;
                float left = box[1] * width;
                float bottom = box[2] * height;
                float right = box[3] * width;

                // Ensure the coordinates are within bounds
                top = Math.max(0, Math.min(height - 1, top));
                left = Math.max(0, Math.min(width - 1, left));
                bottom = Math.max(0, Math.min(height - 1, bottom));
                right = Math.max(0, Math.min(width - 1, right));

                Imgproc.rectangle(mat_image, new Point(left, top), new Point(right, bottom), new Scalar(0, 255, 0, 255), 2);
                Imgproc.putText(mat_image, labelList.get((int) box[4]) + " " + confidencePercentage + "%", new Point(left, top), 3, 1, new Scalar(0, 255, 0, 255), 2);
            }
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
    private ByteBuffer convertMatToByteBuffer(Mat mat_image) {
        // Resize the image using OpenCV functions
        Imgproc.resize(mat_image, mat_image, new Size(INPUT_SIZE, INPUT_SIZE));

        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4); // Assuming 3 channels, 4 bytes per float
        byteBuffer.order(ByteOrder.nativeOrder());

        for (int row = 0; row < INPUT_SIZE; row++) {
            for (int col = 0; col < INPUT_SIZE; col++) {
                double[] pixel = mat_image.get(row, col);
                byteBuffer.putFloat((float) (pixel[0] / 255.0)); // Assuming pixel values are in the range [0, 255]
                byteBuffer.putFloat((float) (pixel[1] / 255.0));
                byteBuffer.putFloat((float) (pixel[2] / 255.0));
            }
        }

        byteBuffer.flip(); // Important: Reset the position to the beginning of the buffer

        return byteBuffer;
    }


    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        int quant = 1;
        int sizeImages = INPUT_SIZE;

        if (quant == 0) {
            byteBuffer = ByteBuffer.allocateDirect(1 * sizeImages * sizeImages * 3);
        } else {
            byteBuffer = ByteBuffer.allocateDirect(4 * sizeImages * sizeImages * 3);
        }

        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[sizeImages * sizeImages];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;

        for (int i = 0; i < sizeImages; ++i) {
            for (int j = 0; j < sizeImages; ++j) {
                final int val = intValues[pixel++];

                if (quant == 0) {
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) ((val) & 0xFF));
                } else {
                    byteBuffer.putFloat(((val >> 16) & 0xFF));
                    byteBuffer.putFloat(((val >> 8) & 0xFF));
                    byteBuffer.putFloat(((val) & 0xFF));
                }
            }
        }

        return byteBuffer;
    }
}
