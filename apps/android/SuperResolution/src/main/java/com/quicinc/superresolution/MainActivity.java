// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.superresolution;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageDecoder;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import com.quicinc.ImageProcessing;
import com.quicinc.tflite.AIHubDefaults;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.security.NoSuchAlgorithmException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.HashSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {
    // UI Elements
    RadioGroup delegateSelectionGroup;
    RadioButton allDelegatesButton;
    RadioButton cpuOnlyButton;
    ImageView selectedImageView;
    TextView inferenceTimeView;
    TextView predictionTimeView;
    Spinner imageSelector, modelSelector;
    Button predictionButton, saveButton, prediction2Button, openButton;
    ActivityResultLauncher<Intent> selectImageResultLauncher;
    private final String fromGalleryImageSelectorOption = "From Gallery";
    private final String notSelectedImageSelectorOption = "Not Selected";
    private final String[] imageSelectorOptions =
            {notSelectedImageSelectorOption,
                    //   fromGalleryImageSelectorOption,
                    "Sample1.jpg",
                    "Sample2.jpg",
            };

    private String[] modelSelectorOptions;

    // Inference Elements
    Bitmap selectedImage = null, resultImage; // Raw image, not resized
    private SuperResolution defaultDelegateUpscaler;
    private SuperResolution cpuOnlyUpscaler;
    private boolean cpuOnlyClassification = false;
    NumberFormat timeFormatter = new DecimalFormat("0.00");
    ExecutorService backgroundTaskExecutor = Executors.newSingleThreadExecutor();
    Handler mainLooperHandler = new Handler(Looper.getMainLooper());

    /**
     * Instantiate the activity on first load.
     * Creates the UI and a background thread that instantiates the upscaler  TFLite model.
     *
     * @param savedInstanceState Saved instance state.
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //
        // UI Initialization
        //
        setContentView(R.layout.main_activity);
        selectedImageView = (ImageView) findViewById(R.id.selectedImageView);
        delegateSelectionGroup = (RadioGroup) findViewById(R.id.delegateSelectionGroup);
        cpuOnlyButton = (RadioButton) findViewById(R.id.cpuOnlyRadio);
        allDelegatesButton = (RadioButton) findViewById(R.id.defaultDelegateRadio);

        imageSelector = (Spinner) findViewById((R.id.imageSelector));
        modelSelector = (Spinner) findViewById((R.id.modelSelector));
        inferenceTimeView = (TextView) findViewById(R.id.inferenceTimeResultText);
        predictionTimeView = (TextView) findViewById(R.id.predictionTimeResultText);
        predictionButton = (Button) findViewById(R.id.runModelButton);
        prediction2Button = (Button) findViewById(R.id.runModel2Button);
        saveButton = (Button) findViewById(R.id.saveButton);
        openButton = (Button) findViewById(R.id.openButton);

        // Setup Image Selector Dropdown
        ArrayAdapter ad = new ArrayAdapter(this, android.R.layout.simple_spinner_item, imageSelectorOptions);
        ad.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        imageSelector.setAdapter(ad);
        imageSelector.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                Log.w("imageSelector", "onItemSelected " + position);
                // Load selected picture from assets
                ((TextView) view).setTextColor(getResources().getColor(R.color.white));
                if (!parent.getItemAtPosition(position).equals(notSelectedImageSelectorOption)) {
                    resultImage = null;
                    if (parent.getItemAtPosition(position).equals(fromGalleryImageSelectorOption)) {
//                        openButton.setClickable(true);
                        Intent i = new Intent();
                        i.setType("image/*");
                        i.setAction(Intent.ACTION_GET_CONTENT);
                        selectImageResultLauncher.launch(i);
                    } else {
//                        openButton.setClickable(false);
                        loadImageFromStringAsync((String) parent.getItemAtPosition(position));
                    }
                } else {
//                    openButton.setClickable(false);
                    displayDefaultImage();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        openButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                resultImage = null;
                Intent i = new Intent();
                i.setType("image/*");
                i.setAction(Intent.ACTION_GET_CONTENT);
                selectImageResultLauncher.launch(i);
            }
        });


        // Setup Model Selector Dropdown
        modelSelectorOptions = getResources().getStringArray(R.array.model_files);
        ArrayAdapter modelAdapter = new ArrayAdapter(this, android.R.layout.simple_spinner_item, modelSelectorOptions);
        modelAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelSelector.setAdapter(modelAdapter);
        modelSelector.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                // Load selected models from assets
                ((TextView) view).setTextColor(getResources().getColor(R.color.white));
                ((TextView) view).setEllipsize(TextUtils.TruncateAt.END);

                // Exit the UI thread and instantiate the model in the background.
                String modelName = parent.getItemAtPosition(position).toString();
                createTFLiteUpscalerAsync(modelName);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });
        // Setup Image Selection from Phone Gallery
        selectImageResultLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                (ActivityResult result) -> {
                    resultImage = null;
                    if (result.getResultCode() == Activity.RESULT_OK &&
                            result.getData() != null &&
                            result.getData().getData() != null) {
                        loadImageFromURIAsync((Uri) (result.getData().getData()));
                    } else {
                        displayDefaultImage();
                    }
                });

        // Setup delegate selection buttons
        delegateSelectionGroup.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.cpuOnlyRadio) {
                if (!cpuOnlyClassification) {
                    this.cpuOnlyClassification = true;
                    clearPredictionResults();
                }
            } else if (checkedId == R.id.defaultDelegateRadio) {
                if (cpuOnlyClassification) {
                    this.cpuOnlyClassification = false;
                    clearPredictionResults();
                }
            } else {
                throw new RuntimeException("A radio button for selected runtime is not implemented");
            }
        });

        // Setup button callback
        predictionButton.setOnClickListener((view) -> updatePredictionDataAsync(false));
        prediction2Button.setOnClickListener((view) -> updatePredictionDataAsync(true));
        saveButton.setOnClickListener((view) -> {
            if (resultImage != null) {
                saveData();
            }
        });

        // Enable image selection
        enableImageSelector();
        enableDelegateSelectionButtons();
    }

    /**
     * Enable or disable UI controls for inference.
     *
     * @param enabled If true, enable the UI. If false, disable the UI.
     */
    void setInferenceUIEnabled(boolean enabled) {
        if (!enabled) {
            inferenceTimeView.setText("-- ms");
            predictionTimeView.setText("-- ms");
            predictionButton.setEnabled(false);
            predictionButton.setAlpha(0.5f);
            prediction2Button.setEnabled(false);
            prediction2Button.setAlpha(0.5f);
            saveButton.setEnabled(false);
            saveButton.setAlpha(0.5f);
            imageSelector.setEnabled(false);
            imageSelector.setAlpha(0.5f);
            modelSelector.setEnabled(false);
            modelSelector.setAlpha(0.5f);
            cpuOnlyButton.setEnabled(false);
            allDelegatesButton.setEnabled(false);
        } else if (cpuOnlyUpscaler != null && defaultDelegateUpscaler != null && selectedImage != null) {
            predictionButton.setEnabled(true);
            predictionButton.setAlpha(1.0f);
            prediction2Button.setEnabled(true);
            prediction2Button.setAlpha(1.0f);
            enableImageSelector();
            enableModelSelector();
            enableDelegateSelectionButtons();
        }
    }

    /**
     * Enable the image selector UI spinner.
     */
    void enableImageSelector() {
        imageSelector.setEnabled(true);
        imageSelector.setAlpha(1.0f);
    }

    /**
     * Enable the model selector UI spinner.
     */
    void enableModelSelector() {
        modelSelector.setEnabled(true);
        modelSelector.setAlpha(1.0f);
    }

    /**
     * Enable the image selector UI radio buttons.
     */
    void enableDelegateSelectionButtons() {
        cpuOnlyButton.setEnabled(true);
        allDelegatesButton.setEnabled(true);
    }

    /**
     * Reset the selected image view to the default image,
     * and enable portions of the inference UI accordingly.
     */
    void displayDefaultImage() {
        setInferenceUIEnabled(false);
        enableImageSelector();
        enableDelegateSelectionButtons();
        clearPredictionResults();
        selectedImageView.setImageResource(R.drawable.ic_launcher_background);
        selectedImage = null;
        resultImage = null;
    }

    /**
     * Clear previous inference results from the UI.
     */
    void clearPredictionResults() {
        if (selectedImage != null) {
            selectedImageView.setImageBitmap(selectedImage);
        }
        resultImage = null;
        inferenceTimeView.setText("-- ms");
        predictionTimeView.setText("-- ms");
    }

    /**
     * Load an image for inference and update the UI accordingly.
     * The image will be loaded asynchronously to the main UI thread.
     *
     * @param imagePath Path to the image relative to the the `assets/images/` folder
     */
    void loadImageFromStringAsync(String imagePath) {
        setInferenceUIEnabled(false);
        // Exit the main UI thread and load the image in the background.
        backgroundTaskExecutor.execute(() -> {
            // Background task
            try (InputStream inputImage = getAssets().open("images/" + imagePath)) {
                selectedImage = BitmapFactory.decodeStream(inputImage);
                // Downscale the image to the size the model supports.
//                int[] inputSize = defaultDelegateUpscaler.getInputWidthHeight();
//                selectedImage = ImageProcessing.resizeAndPadMaintainAspectRatio(selectedImage, inputSize[0], inputSize[1], 0xFF);
            } catch (IOException e) {
                throw new RuntimeException(e.getMessage());
            }

            mainLooperHandler.post(() -> {
                // In main UI thread
                selectedImageView.setImageBitmap(selectedImage);
                setInferenceUIEnabled(true);
            });
        });
    }

    /**
     * Load an image for inference and update the UI accordingly.
     * The image will be loaded asynchronously to the main UI thread.
     *
     * @param imageUri URI to the image.
     */
    void loadImageFromURIAsync(Uri imageUri) {
        setInferenceUIEnabled(false);
        // Exit the main UI thread and load the image in the background.
        backgroundTaskExecutor.execute(() -> {
            // Background task
            try {
                if (android.os.Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    selectedImage = ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(), imageUri), (decoder, info, src) -> {
                        decoder.setMutableRequired(true);
                    });
                } else {
                    selectedImage = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                }
                // Downscale the image to the size the model supports.
//                int[] inputSize = defaultDelegateUpscaler.getInputWidthHeight();
//                selectedImage = ImageProcessing.resizeAndPadMaintainAspectRatio(selectedImage, inputSize[0], inputSize[1], 0xFF);
            } catch (IOException e) {
                throw new RuntimeException(e.getMessage());
            }

            mainLooperHandler.post(() -> {
                // In main UI thread
                selectedImageView.setImageBitmap(selectedImage);
                setInferenceUIEnabled(true);
            });
        });
    }

    /**
     * Run the upscaler on the currently selected image.
     * Prediction will run asynchronously to the main UI thread.
     * Disables inference UI before inference and re-enables it afterwards.
     */
    void updatePredictionDataAsync(boolean save) {
        setInferenceUIEnabled(false);
        resultImage = null;

        SuperResolution imageClassification;
        if (cpuOnlyClassification) {
            imageClassification = cpuOnlyUpscaler;
        } else {
            imageClassification = defaultDelegateUpscaler;
        }

        // Exit the main UI thread and execute the model in the background.
        backgroundTaskExecutor.execute(() -> {
            // Background task
            long upscaleStartTime = System.nanoTime();
            resultImage = imageClassification.generateUpscaledBigImage(selectedImage);
            long inferenceTime = imageClassification.getInferenceTime();
            String inferenceTimeText = timeFormatter.format((double) inferenceTime / 1000000);
            String predictionTimeText = timeFormatter.format((double) (System.nanoTime() - upscaleStartTime) / 1000000);

            mainLooperHandler.post(() -> {
                // In main UI thread
                selectedImageView.setImageBitmap(resultImage);
                if (save) {
                    saveData();
                }
                inferenceTimeView.setText(inferenceTimeText + " ms");
                predictionTimeView.setText(predictionTimeText + " ms");
                setInferenceUIEnabled(true);
                saveButton.setEnabled(true);
                saveButton.setAlpha(1.0f);
            });
        });
    }

    void saveData() {
        setInferenceUIEnabled(false);

        try {

            ContentValues values = new ContentValues();

            values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
//                values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");

            Uri dataUri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
            Uri fileUri = getContentResolver().insert(dataUri, values);

            if (fileUri == null) {
                return;
            }

            OutputStream outStream = getContentResolver().openOutputStream(fileUri);

            resultImage.compress(Bitmap.CompressFormat.PNG, 100, outStream);
//                resultImage.compress(Bitmap.CompressFormat.JPEG, 90, outStream);
            outStream.flush();
            outStream.close();


            // 刷新相册
            sendBroadcast(new Intent("com.android.camera.NEW_PICTURE", fileUri));
            Toast.makeText(this, "Saved", Toast.LENGTH_SHORT).show();

        } catch (Exception ex) {
            ex.printStackTrace();
            Toast.makeText(this, "Save fail", Toast.LENGTH_SHORT).show();
        }

        setInferenceUIEnabled(true);
    }


    /**
     * Create inference upscaler objects.
     * Loading the TF Lite model takes time, so this is done asynchronously to the main UI thread.
     * Disables the inference UI during load and reenables it afterwards.
     */
    void createTFLiteUpscalerAsync(final String tfLiteModelAsset) {
        if (defaultDelegateUpscaler != null || cpuOnlyUpscaler != null) {
            defaultDelegateUpscaler.close();
            cpuOnlyUpscaler.close();
//            throw new RuntimeException("Classifiers were already created");
        }
        setInferenceUIEnabled(false);

        // Exit the UI thread and instantiate the model in the background.
        backgroundTaskExecutor.execute(() -> {
            // Create two upscalers.
            // One uses the default set of delegates (can access NPU, GPU, CPU), and the other uses only XNNPack (CPU).
            try {
                long modelLoadStartTime = System.nanoTime();
                defaultDelegateUpscaler = new SuperResolution(
                        this,
                        tfLiteModelAsset,
                        AIHubDefaults.delegatePriorityOrder /* AI Hub Defaults */
                );
                long modelLoadTime1 = System.nanoTime();
                cpuOnlyUpscaler = new SuperResolution(
                        this,
                        tfLiteModelAsset,
                        AIHubDefaults.delegatePriorityOrderForDelegates(new HashSet<>() /* No delegates; cpu only */)
                );
                long modelLoadTime2 = System.nanoTime();
                Log.w("model Load time", "default = " + (modelLoadTime1 - modelLoadStartTime) / 1000000
                        + "ms, cpu = " + (modelLoadTime2 - modelLoadTime1) / 1000000 + "ms");
            } catch (IOException | NoSuchAlgorithmException e) {
                throw new RuntimeException(e.getMessage());
            }
            Log.i("createTFLiteUpscalerAsync", "model load finish: " + tfLiteModelAsset);

            mainLooperHandler.post(() -> setInferenceUIEnabled(true));
        });
    }

    /**
     * Destroy this activity and release memory used by held objects.
     */
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cpuOnlyUpscaler != null) cpuOnlyUpscaler.close();
        if (defaultDelegateUpscaler != null) defaultDelegateUpscaler.close();
    }
}
