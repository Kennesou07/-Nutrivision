package com.capstone.nutrivision;
import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.widget.ImageView;

import com.bumptech.glide.Glide;

public class Splashscreen extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.splashscreen);

        ImageView imageView = findViewById(R.id.imageView);

        // Load the GIF into the ImageView using Glide
        Glide.with(this)
                .asGif()
                .load(R.drawable.nutrivision)
                .into(imageView);

        // Set a delay to redirect to the next activity
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                Intent intent = new Intent(Splashscreen.this, Login.class);
                startActivity(intent);
                finish();
            }
        }, 5000);
    }
}