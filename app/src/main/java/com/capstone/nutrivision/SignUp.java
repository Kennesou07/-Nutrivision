package com.capstone.nutrivision;

import static android.widget.Toast.LENGTH_LONG;
import static com.capstone.nutrivision.API.*;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.Adapter;
import android.widget.ArrayAdapter;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import com.google.android.material.snackbar.Snackbar;
import android.view.Gravity;

import com.android.volley.AuthFailureError;
import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.google.android.material.textfield.MaterialAutoCompleteTextView;
import com.google.android.material.textfield.TextInputEditText;

import java.util.HashMap;
import java.util.Map;


public class SignUp extends Activity {
    TextView passClassifier,rePassClassifier,usernameClassifier,emailClassifier;
    TextInputEditText edtUsername, edtPassword, edtRePass,edtName,edtAge,edtEmail;
    Spinner edtGender;
    Button btnSignUp;
    ProgressDialog progressDialog;
    InputMethodManager imm;

    public class CustomSpinnerAdapter extends ArrayAdapter<CharSequence> {
        public CustomSpinnerAdapter(Context context, int resource, CharSequence[] objects) {
            super(context, resource, objects);
        }

        @Override
        public boolean isEnabled(int position) {
            return position != 0; // Disable the first item in the Spinner
        }

        @Override
        public View getDropDownView(int position, View convertView, ViewGroup parent) {
            View view = super.getDropDownView(position, convertView, parent);
            TextView tv = (TextView) view;
            if (position == 0) {
                tv.setTextColor(Color.GRAY); // Set the color of the first item to gray to indicate that it's disabled
            } else {
                tv.setTextColor(Color.BLACK);
            }
            return view;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.signup);
        imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
        edtUsername = findViewById(R.id.editTextUsername);
        edtPassword = findViewById(R.id.editTextPassword);
        edtRePass = findViewById(R.id.editTextRePassword);
        edtName = findViewById(R.id.editTextName);
        edtAge = findViewById(R.id.editTextAge);
        edtGender = findViewById(R.id.spinnerGender);
        edtEmail = findViewById(R.id.editTextEmail);
        passClassifier = findViewById(R.id.passClassifier);
        rePassClassifier = findViewById(R.id.rePassClassifier);
        usernameClassifier = findViewById(R.id.usernameClassifier);
        emailClassifier = findViewById(R.id.emailClassifier);
        btnSignUp = findViewById(R.id.SignUpBtn);
        progressDialog = new ProgressDialog(this);
        progressDialog.setMessage("Loading...");
        progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);

        // Spinner initialization
        edtGender = findViewById(R.id.spinnerGender);
        CharSequence[] items = getResources().getStringArray(R.array.genders);
        CustomSpinnerAdapter adapter = new CustomSpinnerAdapter(this, android.R.layout.simple_spinner_item, items);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        edtGender.setAdapter(adapter);

        edtPassword.setOnFocusChangeListener(new View.OnFocusChangeListener() {
            @Override
            public void onFocusChange(View v, boolean hasFocus) {
                if (hasFocus) {
                    showReminderMessage(v);
                }
            }

            private void showReminderMessage(View view) {
                Snackbar snackbar = Snackbar.make(view, "Password should contain at least 1 uppercase letter and 1 special character", Snackbar.LENGTH_LONG);
                View snackbarView = snackbar.getView();
                FrameLayout.LayoutParams params = (FrameLayout.LayoutParams) snackbarView.getLayoutParams();
                params.gravity = Gravity.TOP;
                snackbarView.setLayoutParams(params);
                snackbar.show();
            }
        });

        edtUsername.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }

            @Override
            public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }

            @Override
            public void afterTextChanged(Editable editable) {
                String username = editable.toString().trim();
                StringRequest stringRequest = new StringRequest(Request.Method.POST, USERNAME_SITE,
                        new Response.Listener<String>() {
                            @Override
                            public void onResponse(String response) {
                                if (response.trim().equals("Exist!")) {
                                    // Registration successful
                                    usernameClassifier.setVisibility(View.VISIBLE);
                                    usernameClassifier.setText("Username already taken!");
                                    usernameClassifier.setTextColor(Color.RED);
                                } else {
                                    // Registration failed
                                    usernameClassifier.setVisibility(View.VISIBLE);
                                    usernameClassifier.setText("Available");
                                    usernameClassifier.setTextColor(Color.GREEN);
                                }
                            }
                        }, new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        progressDialog.dismiss();
                        message(error.getMessage());
                    }
                }){
                    @Override
                    protected Map<String, String> getParams() throws AuthFailureError {
                        Map<String,String> params = new HashMap<>();
                        params.put("Username",username);
                        return params;
                    }
                };
                stringRequest.setRetryPolicy(new DefaultRetryPolicy(1000,
                        DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                        DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
                RequestQueue queue = Volley.newRequestQueue(SignUp.this);
                queue.add(stringRequest);
            }
        });
        edtPassword.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {
                // This method is called to notify you that characters within s are about to be replaced with new text with a length of before.
            }

            @Override
            public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {
                // This method is called to notify you that somewhere within s, the text has been changed.
            }

            @Override
            public void afterTextChanged(Editable editable) {
                // This method is called to notify you that somewhere within editable, the text has been changed.
                String password = editable.toString().trim();
                if (!isPasswordStrong(password)) {
                    passClassifier.setVisibility(View.VISIBLE);
                    passClassifier.setText("Weak Password!");
                    passClassifier.setTextColor(Color.RED);
                } else {
                    passClassifier.setVisibility(View.VISIBLE);
                    passClassifier.setTextColor(Color.GREEN);
                    passClassifier.setText("Strong Password!");
                }
            }
        });

        edtRePass.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }

            @Override
            public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }

            @Override
            public void afterTextChanged(Editable editable) {
                String password = editable.toString().trim();
                String repassword = editable.toString().trim();
                if (!isPasswordMatch(password,repassword)) {
                    rePassClassifier.setVisibility(View.VISIBLE);
                    rePassClassifier.setText("Password Mismatch!");
                    rePassClassifier.setTextColor(Color.RED);
                } else {
                    rePassClassifier.setVisibility(View.VISIBLE);
                    rePassClassifier.setTextColor(Color.GREEN);
                    rePassClassifier.setText("Password Matched!");
                }
            }
        });

        btnSignUp.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                progressDialog.show();
                final String username = edtUsername.getText().toString().trim();
                final String password = edtPassword.getText().toString().trim();
                final String repassword = edtRePass.getText().toString().trim();
                final String name = edtName.getText().toString().trim();
                final String age = edtAge.getText().toString().trim();
                final String gender = edtGender.getSelectedItem().toString();
                final String email = edtEmail.getText().toString().trim();

                if(username.isEmpty() || password.isEmpty() || repassword.isEmpty() || name.isEmpty() || age.isEmpty() || gender.isEmpty() || email.isEmpty()){
                    message("Please input all fields...");
                    progressDialog.dismiss();
                }
                else{
                    if(isPasswordMatch(password,repassword) && isPasswordStrong(password)){
                        passClassifier.setVisibility(View.INVISIBLE);
                        rePassClassifier.setVisibility(View.INVISIBLE);
                        StringRequest stringRequest = new StringRequest(Request.Method.POST, SIGNUP_SITE,
                                new Response.Listener<String>() {
                                    @Override
                                    public void onResponse(String response) {
                                        progressDialog.dismiss();
                                        Log.d("ServerSignUp", "Server Response: " + response);
                                        if (response.trim().equals("Success!")) {
                                            // Registration successful
                                            message("Registration Successful!");
                                            Intent intent = new Intent(SignUp.this,Login.class);
                                            startActivity(intent);
                                            finish();
                                        } else {
                                            // Registration failed
                                            message(response);
                                            edtEmail.requestFocus();
                                            imm.showSoftInput(edtEmail,InputMethodManager.SHOW_IMPLICIT);
                                            emailClassifier.setVisibility(View.VISIBLE);
                                            emailClassifier.setText(response);
                                            emailClassifier.setTextColor(Color.RED);
                                        }
                                    }
                                }, new Response.ErrorListener() {
                            @Override
                            public void onErrorResponse(VolleyError error) {
                                progressDialog.dismiss();
                                Log.e("SignUp", "Volley Error: " + error.toString());
                                message(error.getMessage());
                            }
                        }){
                            @Override
                            protected Map<String, String> getParams() throws AuthFailureError {
                                Map<String,String> params = new HashMap<>();

                                params.put("Username",username);
                                params.put("Password",password);
                                params.put("Name",name);
                                params.put("Age",age);
                                params.put("Gender",gender);
                                params.put("Email",email);
                                return params;
                            }
                        };
                        stringRequest.setRetryPolicy(new DefaultRetryPolicy(20000,
                                DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                                DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
                        RequestQueue queue = Volley.newRequestQueue(SignUp.this);
                        queue.add(stringRequest);
                    }
                    else{
                        if(!isPasswordMatch(password,repassword)){
                            rePassClassifier.setVisibility(View.VISIBLE);
                            rePassClassifier.setText("Password Mismatch!");
                            rePassClassifier.setTextColor(Color.RED);
                            edtRePass.requestFocus();
                            imm.showSoftInput(edtRePass,InputMethodManager.SHOW_IMPLICIT);
                            progressDialog.dismiss();
                            return;
                        }
                        else if(!isPasswordStrong(password)){
                            passClassifier.setVisibility(View.VISIBLE);
                            passClassifier.setText("Weak Password");
                            passClassifier.setTextColor(Color.RED);
                            edtPassword.requestFocus();
                            imm.showSoftInput(edtPassword,InputMethodManager.SHOW_IMPLICIT);
                            progressDialog.dismiss();
                            return;
                        }
                    }
                }
            }
        });
    }
    public void message(String message){
        Toast.makeText(this,message, LENGTH_LONG).show();
    }
    private boolean isPasswordStrong(String password) {
        // Check for at least 8 characters
        if (password.length() < 8) {
            return false;
        }

        // Check for at least one uppercase letter
        if (!password.matches(".*[A-Z].*")) {
            return false;
        }

        // Check for at least one digit
        if (!password.matches(".*\\d.*")) {
            return false;
        }

        // Check for at least one symbol (special character)
        if (!password.matches(".*[!@#$%^&*()_+\\-=\\[\\]{};':\",./<>?].*")) {
            return false;
        }
        return true;
    }
    private boolean isPasswordMatch(String password, String repassword){
        // Use equals method to compare the text content of the EditTexts
        password = edtPassword.getText().toString().trim();
        repassword = edtRePass.getText().toString().trim();
        return password.equals(repassword);
    }
}