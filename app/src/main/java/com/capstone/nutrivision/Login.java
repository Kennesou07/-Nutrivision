package com.capstone.nutrivision;

import static android.widget.Toast.LENGTH_SHORT;

import static com.capstone.nutrivision.API.*;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.text.Layout;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.google.android.gms.auth.api.signin.GoogleSignIn;
import com.google.android.gms.auth.api.signin.GoogleSignInAccount;
import com.google.android.gms.auth.api.signin.GoogleSignInClient;
import com.google.android.gms.auth.api.signin.GoogleSignInOptions;
import com.google.android.gms.auth.api.signin.GoogleSignInStatusCodes;
import com.google.android.gms.common.api.ApiException;
import com.google.android.gms.tasks.Task;
import com.google.android.material.textfield.MaterialAutoCompleteTextView;
import com.google.android.material.textfield.TextInputEditText;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

public class Login extends Activity {
    GoogleSignInOptions gso;
    GoogleSignInClient gsc;
    ImageView googleBtn;
    Button login,sendEmail;
    TextView forgotPass, register,btnSignUp;
    TextInputEditText editTextUsername,editTextPassword;
    EditText editTextEmail;
    ProgressDialog progressDialog;
    InputMethodManager imm;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.login);
        checkSession();
        gso = new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN).requestEmail().build();
        gsc = GoogleSignIn.getClient(this,gso);
        GoogleSignInAccount acc = GoogleSignIn.getLastSignedInAccount(this);

        if(acc != null){
            goToLandingPage();
        }
        googleBtn = findViewById(R.id.googleBtn);
        imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
        editTextUsername = findViewById(R.id.emailEditText);
        editTextPassword = findViewById(R.id.passwordEditText);
        progressDialog = new ProgressDialog(this);
        progressDialog.setMessage("Logging in");
        progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);

        login = findViewById(R.id.loginButton);
        login.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String username = editTextUsername.getText().toString().trim();
                String password = editTextPassword.getText().toString().trim();
                if(username.isEmpty()){
                    editTextUsername.requestFocus();
                    imm.showSoftInput(editTextUsername,InputMethodManager.SHOW_IMPLICIT);
                }
                else if(password.isEmpty()){
                    editTextPassword.requestFocus();
                    imm.showSoftInput(editTextPassword,InputMethodManager.SHOW_IMPLICIT);
                }
                else{
                    StringRequest stringRequest = new StringRequest(Request.Method.POST, LOGIN_SITE,
                            new Response.Listener<String>() {
                                @Override
                                public void onResponse(String response) {
                                    try {
                                        JSONObject object = new JSONObject(response);
                                        String result = object.getString("status");
                                        if(result.equals("success")){
                                            progressDialog.dismiss();

                                            // Save session information
                                            SharedPreferences preferences = getSharedPreferences("LogInSession", MODE_PRIVATE);
                                            SharedPreferences.Editor editor = preferences.edit();
                                            editor.putString("username", username);
                                            editor.putBoolean("isLoggedIn", true);
                                            editor.apply();

                                            Toast.makeText(Login.this,"Login Success! Please wait.",LENGTH_SHORT).show();
                                            Intent intent = new Intent(Login.this,MainActivity.class);
                                            startActivity(intent);
                                            finish();
                                        }
                                        else{
                                            progressDialog.dismiss();
                                            Toast.makeText(Login.this,"Invalid Credentials!", LENGTH_SHORT).show();
                                        }
                                    } catch (JSONException e) {
                                        throw new RuntimeException(e);
                                    }

                                }
                            }, new Response.ErrorListener() {
                        @Override
                        public void onErrorResponse(VolleyError error) {
                            progressDialog.dismiss();
                        }
                    }){
                        @Override
                        protected Map<String, String> getParams() throws AuthFailureError {
                            Map<String,String> params = new HashMap<>();
                            params.put("Username",username);
                            params.put("Password",password);
                            return params;
                        }
                    };
                    stringRequest.setRetryPolicy(new DefaultRetryPolicy(1000,
                            DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                            DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
                    RequestQueue queue = Volley.newRequestQueue(Login.this);
                    queue.add(stringRequest);
                }
            }
        });
        forgotPass = findViewById(R.id.forgotPass);
        forgotPass.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showForgotPasswordDialog();
            }
        });
        register = findViewById(R.id.registerTextView);
        register.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Login.this,SignUp.class);
                startActivity(intent);
            }
        });
        googleBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent googleSignIn = gsc.getSignInIntent();
                startActivityForResult(googleSignIn,1000);

            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == 1000){
            Task<GoogleSignInAccount> task = GoogleSignIn.getSignedInAccountFromIntent(data);
            handleSignInResult(task);
        }
    }
    private void handleSignInResult(Task<GoogleSignInAccount> completedTask) {
        try {
            GoogleSignInAccount account = completedTask.getResult(ApiException.class);
            Intent intent = new Intent(Login.this,MainActivity.class);
            startActivity(intent);
            finish();
            // Signed in successfully, show authenticated UI.
        } catch (ApiException e) {
            // The ApiException status code indicates the detailed failure reason.
            // Please refer to the GoogleSignInStatusCodes class reference for more information.
            Log.w("GoogleSignIn", "signInResult:failed code=" + e.getStatusCode());
        }
    }

    private void checkSession(){
        SharedPreferences preferences = getSharedPreferences("LogInSession", MODE_PRIVATE);
        boolean isLoggedIn = preferences.getBoolean("isLoggedIn", false);

        if (isLoggedIn) {
            goToLandingPage();
        }
    }
    private void goToLandingPage(){
        Intent intent = new Intent(Login.this, MainActivity.class);
        startActivity(intent);
        finish();
    }

    private void showForgotPasswordDialog(){
        LayoutInflater inflater = (LayoutInflater) getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        View dialogView = inflater.inflate(R.layout.forgotpassword,null);
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setView(dialogView);
        final AlertDialog alertDialog = builder.create();
        editTextEmail = dialogView.findViewById(R.id.edtTextEmail);
        sendEmail = dialogView.findViewById(R.id.btnSendEmail);

        sendEmail.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String email = editTextEmail.getText().toString().trim();
                if (email == null) {
                    Toast.makeText(Login.this,"Email is null",LENGTH_SHORT).show();
                    editTextEmail.requestFocus();
                    imm.showSoftInput(editTextEmail,InputMethodManager.SHOW_IMPLICIT);
                }
                else{
                    String resetToken = TokenGenerator.generateToken();
                    StringRequest stringRequest = new StringRequest(Request.Method.POST, FORGOT_SITE,
                            new Response.Listener<String>() {
                                @Override
                                public void onResponse(String response) {
                                    try {
                                        JSONObject jsonResponse = new JSONObject(response);
                                        String status = jsonResponse.getString("Status");
                                        String message = jsonResponse.getString("Message");
                                        if ("Success".equals(status)) {
                                            Toast.makeText(Login.this, message, Toast.LENGTH_SHORT).show();
                                        } else {
                                            Toast.makeText(Login.this, message, Toast.LENGTH_SHORT).show();
                                        }
                                    } catch (JSONException e) {
                                        e.printStackTrace();
                                        Toast.makeText(Login.this, "JSON parsing error", Toast.LENGTH_SHORT).show();
                                    }
                                }
                            }, new Response.ErrorListener() {
                        @Override
                        public void onErrorResponse(VolleyError error) {
                            progressDialog.dismiss();
                        }
                    }){
                        @Override
                        protected Map<String, String> getParams() throws AuthFailureError {
                            Map<String,String> params = new HashMap<>();
                            params.put("Email",email);
                            params.put("Token",resetToken);
                            return params;
                        }
                    };
                    stringRequest.setRetryPolicy(new DefaultRetryPolicy(1000,
                            DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                            DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
                    RequestQueue queue = Volley.newRequestQueue(Login.this);
                    queue.add(stringRequest);
                }
                alertDialog.dismiss();
            }
        });
        alertDialog.show();
    }
}