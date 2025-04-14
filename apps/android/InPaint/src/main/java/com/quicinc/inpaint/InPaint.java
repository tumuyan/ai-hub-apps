// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.inpaint;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.util.Log;
import android.util.Pair;

import com.quicinc.ImageProcessing;
import com.quicinc.tflite.AIHubDefaults;
import com.quicinc.tflite.TFLiteHelpers;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ColorSpaceType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class InPaint implements AutoCloseable {
    private static final String TAG = "InPaint";
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;
    private final int[] inputShape;
    private int scale=1;
    private final DataType inputType;
    private final DataType outputType;
    private long preprocessingTime;
    private long postprocessingTime;
    private final ImageProcessor inputImageProcessor;
    private final ImageProcessor maskImageProcessor;
    private final ImageProcessor outputImageProcessor;
    private final TensorBuffer outputBuffer;
    private final Map<Integer, Object> outputBindings;
    private final TensorImage outputImage;

    /**
     * Create an Image Classifier from the given model.
     * Uses default compute units: NPU, GPU, CPU.
     * Ignores compute units that fail to load.
     *
     * @param context    App context.
     * @param modelPath  Model path to load.
     * @throws IOException If the model can't be read from disk.
     */
    public InPaint(Context context,
                   String modelPath) throws IOException, NoSuchAlgorithmException {
        this(context, modelPath, AIHubDefaults.delegatePriorityOrder);
    }

    /**
     * Create an Image Classifier from the given model.
     * Ignores compute units that fail to load.
     *
     * @param context     App context.
     * @param modelPath   Model path to load.
     * @param delegatePriorityOrder Priority order of delegate sets to enable.
     * @throws IOException If the model can't be read from disk.
     */
    public InPaint(Context context,
                   String modelPath,
                   TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {
        // Load TF Lite model
        Pair<MappedByteBuffer, String> modelAndHash = TFLiteHelpers.loadModelFile(context.getAssets(), modelPath);
        Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                modelAndHash.first,
                delegatePriorityOrder,
                AIHubDefaults.numCPUThreads,
                context.getApplicationInfo().nativeLibraryDir,
                context.getCacheDir().getAbsolutePath(),
                modelAndHash.second
        );
        tfLiteInterpreter = iResult.first;
        tfLiteDelegateStore = iResult.second;

        Log.w("ModelInfo", "input count:"+tfLiteInterpreter.getInputTensorCount()+", output count"+tfLiteInterpreter.getOutputTensorCount());

        // Validate TF Lite model fits requirements for this app
        assert tfLiteInterpreter.getInputTensorCount() == 2;
        Tensor inputTensor = tfLiteInterpreter.getInputTensor(0);
        Tensor maskTensor = tfLiteInterpreter.getInputTensor(1);
        inputShape = inputTensor.shape();
        inputType = inputTensor.dataType();
        int[] maskShape = maskTensor.shape();

        assert maskTensor.dataType() == inputType;
        assert maskShape.length == 4; // 4D Mask Tensor: [Batch, Height, Width, Channels]
        assert maskShape[0] == 1; // Batch size is 1
        assert maskShape[1] == inputShape[1];
        assert maskShape[2] == inputShape[2];
        assert maskShape[3] == 1; // Mask tensor should have 1 channel

        Log.w("ModelInfo","Model path = "+modelPath
                +", inputShape = "+ Arrays.toString(Arrays.stream(inputShape).toArray()) +", length = "+inputShape.length
                +", maskShape = "+Arrays.toString(Arrays.stream(maskShape).toArray())
        );
        assert inputShape.length == 4; // 4D Input Tensor: [Batch, Height, Width, Channels]
        assert inputShape[0] == 1; // Batch size is 1
        assert inputShape[3] == 3; // Input tensor should have 3 channels
        assert inputType == DataType.UINT8 || inputType == DataType.FLOAT32; // UINT8 (Quantized) and FP32 Input Supported

        assert tfLiteInterpreter.getOutputTensorCount() == 1;
        Tensor outputTensor = tfLiteInterpreter.getOutputTensor(0);
        int []outputShape = outputTensor.shape();
        outputType = outputTensor.dataType();
        Log.w("ModelInfo","Model path = "+modelPath+", outputShape = "+ Arrays.toString(Arrays.stream(outputShape).toArray()) +", length = "+outputShape.length);
        assert outputShape.length == 4; // 4D Output Tensor: [Batch, Height, Width, Channels]
        assert outputShape[0] == 1; // Batch size is 1
        assert outputShape[3] == 3; // Output tensor should have 3 channels
        assert outputType == DataType.UINT8 || inputType == DataType.FLOAT32; // UINT8 (Quantized) and FP32 Input Supported
        scale = outputShape[1]/inputShape[1];
        // Set-up preprocessor
        inputImageProcessor = new ImageProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build();
        maskImageProcessor = new ImageProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build();
        outputImageProcessor = new ImageProcessor.Builder().add(new NormalizeOp(0.0f, 1 / 255.0f)).add(new CastOp(DataType.UINT8)).build();

        // Set-up output image
        outputBuffer = TensorBuffer.createFixedSize(outputShape, outputType);
        outputBindings = new HashMap<>();
        outputBindings.put(0, outputBuffer.getBuffer());
        outputImage = new TensorImage(outputType);
        outputImage.load(outputBuffer, ColorSpaceType.RGB);
    }

    /**
     * Free resources used by the classifier.
     */
    @Override
    public void close() {
        tfLiteInterpreter.close();
        for (Delegate delegate: tfLiteDelegateStore.values()) {
            delegate.close();
        }
    }

    /**
     * @return last preprocessing time in microseconds.
     */
    public long getLastPreprocessingTime() {
        if (preprocessingTime == 0) {
            throw new RuntimeException("Cannot get preprocessing time as model has not yet been executed.");
        }
        return preprocessingTime;
    }

    /**
     * @return last inference time in microseconds.
     */
    public long getLastInferenceTime() {
        return tfLiteInterpreter.getLastNativeInferenceDurationNanoseconds();
    }

    /**
     * @return last postprocessing time in microseconds.
     */
    public long getLastPostprocessingTime() {
        if (postprocessingTime == 0) {
            throw new RuntimeException("Cannot get postprocessing time as model has not yet been executed.");
        }
        return postprocessingTime;
    }

    /** Model input height and width. **/
    public int[] getInputWidthHeight() {
        return new int[] {inputShape[1], inputShape[2]};
    }

    /**
     * Preprocess using the provided image (resize, convert to model input data type).
     * Sets the input buffer held by this.tfLiteModel to the processed input.
     *
     * @param image RGBA-8888 Bitmap to preprocess.
     * @return Array of inputs to pass to the interpreter.
     */
    private ByteBuffer preprocess(Bitmap image,int channel) {
        long prepStartTime = System.nanoTime();
        Bitmap resizedImg;

        // Resize input image
        if (image.getWidth() > inputShape[1] || image.getHeight() > inputShape[2]) {
            resizedImg = ImageProcessing.resizeAndPadMaintainAspectRatio(image, inputShape[1], inputShape[2], 0xFF,channel);
            // This image is larger than the model's desired input size.
            // While this app could easily resize the large image to fit, that defeats the purpose of super resolution.
//            throw new RuntimeException("Input image (" + image.getHeight()  + "*" +image.getWidth() + ") is too big for this model. Expected Width of " + inputShape[1] + " and Height of " + inputShape[2]);
        } else  {
            resizedImg = ImageProcessing.padding(image, inputShape[1], inputShape[2], 0xFF,channel);
        }
        ByteBuffer inputBuffer;
        if (image.getConfig()==Bitmap.Config.ALPHA_8){
            inputBuffer = bitmapToByteBuffer(resizedImg);
            Log.d("Image","Width="+resizedImg.getWidth()+", Height="+resizedImg.getHeight()+", channel="+resizedImg.getConfig().toString()+", buff="+inputBuffer.array().length);
          return   inputBuffer;
        }

        // Convert type and fill input buffer

        TensorImage tImg = TensorImage.fromBitmap(resizedImg);
        int[] shape = tImg.getTensorBuffer().getShape();
        if (inputType == DataType.FLOAT32) {
            // Divide float values by 255
            inputBuffer = inputImageProcessor.process(tImg).getBuffer();
        } else {
            inputBuffer = tImg.getTensorBuffer().getBuffer();
        }
        Log.w("Image","channel="+channel+", config="+image.getConfig() + ", tImg shape("+shape.length+")="+ shape[0]+","+shape[1]+","+shape[2]+", length="+inputBuffer.array().length);

        preprocessingTime = System.nanoTime() - prepStartTime;
        Log.d(TAG, "Preprocessing Time: " + preprocessingTime / 1000000 + " ms");

        return inputBuffer;
    }


    public static ByteBuffer bitmapToByteBuffer(Bitmap bitmap) {

        // 创建 ByteBuffer，假设每个像素使用 4 字节（浮点数）
        int batchSize = 1;
        int height = 512;
        int width = 512;
        int channels = 1; // 单通道

        // 计算字节大小
        int byteSize = batchSize * height * width * channels * 4; // 4 字节
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(byteSize);

        // 将 Bitmap 的像素复制到 ByteBuffer 中
        bitmap.copyPixelsToBuffer(byteBuffer);

        // 重置 ByteBuffer 的位置，以便后续读取
        byteBuffer.rewind();

        return byteBuffer;
    }


    /**
     * Reads the output buffer on tfLiteModel and processes it into a readable output image.
     *
     * @return Upscaled image, in RGBA-8888 format.
     */
    private Bitmap postprocess() {
        long postStartTime = System.nanoTime();

        TensorImage img = outputImage;
        if (outputType == DataType.FLOAT32) {
            // Multiply float values by 255
            img = outputImageProcessor.process(outputImage);
        }
        Bitmap bitmap = img.getBitmap();

        postprocessingTime = System.nanoTime() - postStartTime;
        Log.d(TAG, "Postprocessing Time: " + postprocessingTime / 1000000 + " ms");

        return bitmap;
    }

    /**
     * Upscale the provided input image.
     *
     * @param image RGBA-8888 bitmap image to upscale.
     * @return Predicted, upscaled image, in RGBA-8888 format.
     */
    public Bitmap generateUpscaledImage(Bitmap image, Bitmap mask) {
        // Preprocessing: Resize, convert type
        ByteBuffer imgBuffer = preprocess(image,3);
        ByteBuffer maskBuffer =  preprocess(mask,1);

        // Inference
        outputBuffer.getBuffer().clear();
        tfLiteInterpreter.runForMultipleInputsOutputs(new ByteBuffer[]{imgBuffer,maskBuffer}, outputBindings);

        // Postprocessing: Compute top K indices and convert to labels
        return postprocess();
    }

    long inferenceTime=0;

    public long getInferenceTime() {
        return inferenceTime;
    }


}
