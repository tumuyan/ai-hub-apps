// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.RectF;

public class ImageProcessing {
    /**
     * Resize a bitmap while respecting its aspect ratio.
     * If the output image cannot fit perfectly within the requested output height / width,
     * padding is added such that the output bitmap will be the requested size.
     *
     * @param image              Image to resize
     * @param outputBitmapWidth  Final width
     * @param outputBitmapHeight Final height
     * @param paddingValue       Value to use for padding (usually 0 or 0xFF)
     * @return Resized & padded bitmap
     */
    public static Bitmap resizeAndPadMaintainAspectRatio(
            Bitmap image,
            int outputBitmapWidth,
            int outputBitmapHeight,
            int paddingValue,
            int channel
    ) {
        int width = image.getWidth();
        int height = image.getHeight();
        float ratioBitmap = (float) width / (float) height;
        float ratioMax = (float) outputBitmapWidth / (float) outputBitmapHeight;

        int finalWidth = outputBitmapWidth;
        int finalHeight = outputBitmapHeight;
        if (ratioMax > ratioBitmap) {
            finalWidth = (int) ((float) outputBitmapHeight * ratioBitmap);
        } else {
            finalHeight = (int) ((float) outputBitmapWidth / ratioBitmap);
        }

        if (channel == 1) {
            Bitmap outputImage = Bitmap.createBitmap(outputBitmapWidth, outputBitmapHeight, Bitmap.Config.ALPHA_8);
            Canvas can = new Canvas(outputImage);
            can.drawARGB(0xFF, paddingValue, paddingValue, paddingValue);
            can.drawBitmap(image, null, new RectF(0, 0, finalWidth, finalHeight), null);
            return outputImage;
        } else {
            Bitmap outputImage = Bitmap.createBitmap(outputBitmapWidth, outputBitmapHeight, Bitmap.Config.ARGB_8888);
            Canvas can = new Canvas(outputImage);
            can.drawARGB(0xFF, paddingValue, paddingValue, paddingValue);
            can.drawBitmap(image, null, new RectF(0, 0, finalWidth, finalHeight), null);
            return outputImage;
        }
    }

    /**
     * padding a bitmap without ratio.
     *
     * @param image              Image to resize
     * @param outputBitmapWidth  Final width
     * @param outputBitmapHeight Final height
     * @param paddingValue       Value to use for padding (usually 0 or 0xFF)
     * @return Resized & padded bitmap
     */
    public static Bitmap padding(
            Bitmap image,
            int outputBitmapWidth,
            int outputBitmapHeight,
            int paddingValue,
            int channel
    ) {
        Bitmap outputImage;
        if (channel == 1)
            outputImage = Bitmap.createBitmap(outputBitmapWidth, outputBitmapHeight, Bitmap.Config.ALPHA_8);
        else
            outputImage = Bitmap.createBitmap(outputBitmapWidth, outputBitmapHeight, Bitmap.Config.ARGB_8888);
        Canvas can = new Canvas(outputImage);
        can.drawARGB(0xFF, paddingValue, paddingValue, paddingValue);
        can.drawBitmap(image, null, new RectF(0, 0, image.getWidth(), image.getHeight()), null);
        return outputImage;
    }

    public static Bitmap cropBitmap(Bitmap original, int x0, int y0, int x1, int y1) {
        // 确保坐标在Bitmap范围内
        if (original == null || x0 < 0 || y0 < 0 || x1 > original.getWidth() || y1 > original.getHeight() || x0 >= x1 || y0 >= y1) {
            throw new IllegalArgumentException("Invalid crop coordinates or bitmap is null");
        }
        // 裁剪区域
        return Bitmap.createBitmap(original, x0, y0, x1 - x0, y1 - y0);
    }
}
