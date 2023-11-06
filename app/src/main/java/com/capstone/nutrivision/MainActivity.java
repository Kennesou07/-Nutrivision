package com.capstone.nutrivision;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

import android.Manifest;
import android.widget.ImageView;

import com.google.android.gms.auth.api.signin.GoogleSignIn;
import com.google.android.gms.auth.api.signin.GoogleSignInAccount;
import com.google.android.gms.auth.api.signin.GoogleSignInClient;
import com.google.android.gms.auth.api.signin.GoogleSignInOptions;
import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    GoogleSignInOptions gso;
    GoogleSignInClient gsc;
    Button select,capture,realTime,signOut;
    private ImageView imgView;
    private Bitmap bitmap,bitmap1;
    Mat selectedImage;
    private objectDetectorClass objectDetectorClass;
    int SELECT_CODE = 100, CAPTURE_CODE = 102, REALTIME_CODE = 103;
    static{
        if(OpenCVLoader.initDebug()){
            Log.d("MainActivity","OpenCv instantiated");
        }
        else{
            Log.d("MainActivity","failed to load");
        }
    }
    void getPermission(){
        if(checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(new String[]{Manifest.permission.CAMERA},100);
            requestPermissions(new String[] {Manifest.permission.CAMERA},103);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode == REALTIME_CODE && grantResults.length>0) {
            if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                getPermission();
            }
        }
        if(requestCode == SELECT_CODE && grantResults.length>0){
            if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                getPermission();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == SELECT_CODE && data != null){
            Uri selectedImageUri = data.getData();
            if(selectedImageUri != null){
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),selectedImageUri);
                }
                catch (IOException e){
                    e.printStackTrace();
                }
                selectedImage = new Mat(bitmap.getHeight(),bitmap.getWidth(), CvType.CV_8UC4);
                Utils.bitmapToMat(bitmap,selectedImage);
                selectedImage = objectDetectorClass.recognizePhoto(selectedImage);
                bitmap1 = Bitmap.createBitmap(selectedImage.cols(),selectedImage.rows(),Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(selectedImage,bitmap1);
                imgView.setImageBitmap(bitmap1);
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getPermission();
        gso = new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN).requestEmail().build();
        gsc = GoogleSignIn.getClient(this,gso);
        GoogleSignInAccount acc = GoogleSignIn.getLastSignedInAccount(this);

        if(acc != null){
            String name = acc.getDisplayName();
            String email = acc.getEmail();

        }
        try{
            //input size for 640 for this model
            objectDetectorClass = new objectDetectorClass(getAssets(),"custom_best_float32.tflite","label.txt",640);
            Log.d("MainActivity","Model is successfully loaded");
        }
        catch(IOException e){

        }
        setContentView(R.layout.activity_main);
        imgView = findViewById(R.id.imgView);
        realTime = findViewById(R.id.realTimeBtn);
        realTime.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this,RealTimeCapture.class).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
                startActivityForResult(intent,REALTIME_CODE);
            }
        });
        select = findViewById(R.id.storageBtn);
        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent,"Select Photo"),SELECT_CODE);
            }
        });
        signOut = findViewById(R.id.signOutBtn);
        signOut.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                logout();
            }
        });
    }
    private void logout() {
        gsc.signOut().addOnCompleteListener(new OnCompleteListener<Void>() {
            @Override
            public void onComplete(@NonNull Task<Void> task) {
                clearPreferences();
                finish();
                startActivity(new Intent(MainActivity.this, Login.class));
            }
        });
    }

    private void clearPreferences() {
        SharedPreferences preferences = getSharedPreferences("LogInSession", MODE_PRIVATE);
        SharedPreferences.Editor editor = preferences.edit();
        editor.clear();
        editor.apply();
    }

}