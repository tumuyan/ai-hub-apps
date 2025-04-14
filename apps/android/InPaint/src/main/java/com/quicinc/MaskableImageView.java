package com.quicinc;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View; // Added for OnLayoutChangeListener
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.AppCompatImageView;

import java.nio.ByteBuffer;

public class MaskableImageView extends AppCompatImageView implements View.OnLayoutChangeListener {

    private Bitmap originalBitmap;
    private Bitmap maskBitmap; // ARGB_8888 for drawing flexibility
    private Canvas maskCanvas;

    private Paint maskDrawingPaint; // Paint for drawing strokes onto the maskCanvas
    private Paint maskOverlayPaint; // Paint for drawing the maskBitmap onto the view's canvas in onDraw

    private Path drawPath;
    private float touchX, touchY;
    private static final float TOUCH_TOLERANCE = 4;

    private boolean showMask = false;
    private boolean isDrawingEnabled = false;

    // Matrix for mapping view coordinates to bitmap coordinates
    private final Matrix viewToBitmapMatrix = new Matrix();
    private final float[] mappedPoints = new float[2];
    private boolean imageMatrixCalculated = false; // Flag to track if matrix is ready

    public MaskableImageView(@NonNull Context context) {
        super(context);
        init();
    }

    public MaskableImageView(@NonNull Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public MaskableImageView(@NonNull Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        // Ensure we get layout changes to update the matrix
        addOnLayoutChangeListener(this);

        drawPath = new Path();

        maskDrawingPaint = new Paint();
        maskDrawingPaint.setAntiAlias(true);
        maskDrawingPaint.setDither(true);
        // Use a solid color for the mask stroke first. The alpha will be handled by the bitmap itself.
        maskDrawingPaint.setColor(Color.RED); // Draw solid red onto the mask bitmap
        maskDrawingPaint.setStyle(Paint.Style.STROKE);
        maskDrawingPaint.setStrokeJoin(Paint.Join.ROUND);
        maskDrawingPaint.setStrokeCap(Paint.Cap.ROUND);
        maskDrawingPaint.setStrokeWidth(5f); // Adjust stroke width as needed

        // Paint used to draw the maskBitmap onto the main canvas in onDraw
        maskOverlayPaint = new Paint(Paint.DITHER_FLAG);
        // Apply semi-transparency when drawing the overlay
        maskOverlayPaint.setAlpha(128); // Make the overlay semi-transparent red
    }

    public void setImage(Bitmap bitmap) {
        setImage(bitmap, false);
    }
    /**
     * Sets the Bitmap to display and controls whether mask drawing is enabled.
     *
     * @param bitmap   The Bitmap to display. Can be null to clear the view.
     * @param showMask If true, enables mask drawing on touch and shows the mask overlay.
     *                 If false, disables drawing and shows only the original bitmap.
     */
    public void setImage(@Nullable Bitmap bitmap, boolean showMask) {
        // Reset matrix calculation flag when image changes
        imageMatrixCalculated = false;

        // Recycle old mask if necessary
        if (maskBitmap != null && !maskBitmap.isRecycled()) {
            // Only recycle if the new bitmap is different or null
            if (bitmap == null || originalBitmap == null || bitmap.getWidth() != originalBitmap.getWidth() || bitmap.getHeight() != originalBitmap.getHeight()) {
                clearMaskInternal();
            }
        }

        this.originalBitmap = bitmap;
        this.showMask = showMask;
        this.isDrawingEnabled = showMask && (originalBitmap != null); // Enable drawing only if mask mode is on AND bitmap exists
        float strokeWidth = 30.0f* originalBitmap.getWidth()/ this.getWidth() ;
        maskDrawingPaint.setStrokeWidth(strokeWidth);

        if (originalBitmap != null) {
            // Set the bitmap for the ImageView to handle scaling etc.
            setImageBitmap(originalBitmap); // This triggers layout and matrix calculation eventually

            // Create mask bitmap if needed (or if size mismatched)
            // Use ARGB_8888 for drawing, we extract alpha later in getMask()
            if (maskBitmap == null || maskBitmap.getWidth() != originalBitmap.getWidth() || maskBitmap.getHeight() != originalBitmap.getHeight()) {
                if (maskBitmap != null && !maskBitmap.isRecycled()) {
                    maskBitmap.recycle();
                }
                try {
                    maskBitmap = Bitmap.createBitmap(originalBitmap.getWidth(), originalBitmap.getHeight(), Bitmap.Config.ARGB_8888);
                    maskCanvas = new Canvas(maskBitmap);
                    maskBitmap.eraseColor(Color.TRANSPARENT); // Start with a fully transparent mask
                } catch (OutOfMemoryError e) {
                    System.err.println("Failed to create mask bitmap: " + e.getMessage());
                    maskBitmap = null;
                    maskCanvas = null;
                    isDrawingEnabled = false; // Disable drawing if mask allocation failed
                }
            }
            // If just toggling showMask to true, ensure mask resources exist
            else if (showMask && maskCanvas == null) {
                maskCanvas = new Canvas(maskBitmap); // Re-link canvas if it was detached
            }

        } else {
            // Clear everything if null bitmap is passed
            setImageBitmap(null);
            clearMaskInternal();
        }

        drawPath.reset();
        updateMatrixIfNeeded(); // Try to update matrix immediately if possible
        invalidate(); // Redraw the view
    }

    /**
     * Clears the drawn mask content.
     */
    public void clearMask() {
        if (maskBitmap != null && maskCanvas != null && !maskBitmap.isRecycled()) {
            maskBitmap.eraseColor(Color.TRANSPARENT); // Clear the mask content
            drawPath.reset();
            invalidate(); // Request redraw to show cleared mask
        }
    }

    // Internal helper to release mask resources
    private void clearMaskInternal() {
        if (maskBitmap != null && !maskBitmap.isRecycled()) {
            maskBitmap.recycle();
        }
        maskBitmap = null;
        maskCanvas = null;
        drawPath.reset();
    }


    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        // 1. Let the ImageView draw the original bitmap (scaled/translated)
        super.onDraw(canvas);

        // 2. If mask mode is enabled and mask exists, draw the mask overlay
        if (showMask && maskBitmap != null && !maskBitmap.isRecycled()) {
            // IMPORTANT: Draw the maskBitmap using the same transformation matrix
            // that the ImageView uses to draw the originalBitmap.
            canvas.save(); // Save current canvas state (includes its matrix)
            // Concatenate the ImageView's matrix to the canvas
            // This ensures the mask aligns perfectly with the displayed originalBitmap
            canvas.concat(getImageMatrix());
            // Draw the maskBitmap at (0,0) *in the transformed coordinate space*.
            canvas.drawBitmap(maskBitmap, 0, 0, maskOverlayPaint);
            canvas.restore(); // Restore canvas state
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        // Only handle touch events if drawing is enabled
        if (!isDrawingEnabled || originalBitmap == null || maskCanvas == null || !imageMatrixCalculated) {
            // Don't process touch if not ready (no bitmap, mask disabled, or matrix not calculated)
            return false;
        }

        float x = event.getX();
        float y = event.getY();

        // Map View coordinates to Bitmap coordinates using the inverse matrix
        mapViewToBitmapCoords(x, y);
        float mappedX = mappedPoints[0];
        float mappedY = mappedPoints[1];

        // Optional: Clamp coordinates to be within bitmap bounds if needed,
        // although drawing outside shouldn't cause crashes, just won't appear.
        // mappedX = Math.max(0, Math.min(mappedX, originalBitmap.getWidth() - 1));
        // mappedY = Math.max(0, Math.min(mappedY, originalBitmap.getHeight() - 1));


        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                // Check if the touch point is actually within the bounds of the displayed bitmap area
                // This prevents drawing starting outside the visible image area after mapping
                if (isPointInsideBitmapBounds(mappedX, mappedY)) {
                    touchDown(mappedX, mappedY);
                    invalidate(); // Request redraw
                    return true; // Consume the event
                }
                return false; // Touch started outside image bounds

            case MotionEvent.ACTION_MOVE:
                // Only draw move if touch started inside
                if (!drawPath.isEmpty()) { // Check if ACTION_DOWN was successful
                    touchMove(mappedX, mappedY);
                    invalidate(); // Request redraw
                    return true; // Consume the event
                }
                return false;

            case MotionEvent.ACTION_UP:
                // Only finish path if touch started inside
                if (!drawPath.isEmpty()) {
                    touchUp();
                    invalidate(); // Request redraw
                    return true; // Consume the event
                }
                return false;
        }
        // Let super handle events like scrolling if needed when not drawing
        return super.onTouchEvent(event);
    }

    private void touchDown(float x, float y) {
        drawPath.reset();
        drawPath.moveTo(x, y);
        touchX = x;
        touchY = y;
    }

    private void touchMove(float x, float y) {
        float dx = Math.abs(x - touchX);
        float dy = Math.abs(y - touchY);
        if (dx >= TOUCH_TOLERANCE || dy >= TOUCH_TOLERANCE) {
            // QuadTo creates a smoother line than lineTo
            drawPath.quadTo(touchX, touchY, (x + touchX) / 2, (y + touchY) / 2);
            touchX = x;
            touchY = y;

            // Draw the path onto the mask bitmap canvas (in bitmap coordinates)
            maskCanvas.drawPath(drawPath, maskDrawingPaint);
            // No need to reset path here, continue the stroke
        }
    }

    private void touchUp() {
        if (!drawPath.isEmpty()) {
            drawPath.lineTo(touchX, touchY); // Ensure the last segment is drawn
            // Draw the completed path onto the mask bitmap canvas
            maskCanvas.drawPath(drawPath, maskDrawingPaint);
        }
        // Reset the path for the next stroke
        drawPath.reset();
    }

    /**
     * Maps touch coordinates from the View's coordinate system to the underlying
     * Bitmap's coordinate system using the inverse of the ImageView's current display matrix.
     * Result is stored in mappedPoints[0] (x) and mappedPoints[1] (y).
     *
     * @param viewX X coordinate in the View.
     * @param viewY Y coordinate in the View.
     */
    private void mapViewToBitmapCoords(float viewX, float viewY) {
        if (!imageMatrixCalculated) {
            updateMatrixIfNeeded(); // Ensure matrix is calculated
            if (!imageMatrixCalculated) { // If still not calculated, return (should not happen after layout)
                mappedPoints[0] = -1;
                mappedPoints[1] = -1;
                return;
            }
        }
        // viewToBitmapMatrix already holds the inverse matrix
        mappedPoints[0] = viewX;
        mappedPoints[1] = viewY;
        viewToBitmapMatrix.mapPoints(mappedPoints); // mappedPoints now holds bitmap coordinates
    }

    /**
     * Checks if a point (in bitmap coordinates) falls within the bitmap's dimensions.
     */
    private boolean isPointInsideBitmapBounds(float bitmapX, float bitmapY) {
        return originalBitmap != null && bitmapX >= 0 && bitmapX < originalBitmap.getWidth() &&
                bitmapY >= 0 && bitmapY < originalBitmap.getHeight();
    }


    /**
     * Calculates and caches the inverse matrix needed for coordinate mapping.
     * Should be called after layout changes or image changes.
     */
    private void updateMatrixIfNeeded() {
        // Check if view has width/height and drawable exists
        if (getWidth() > 0 && getHeight() > 0 && getDrawable() != null) {
            // Get the current matrix used by ImageView to display the drawable
            Matrix currentImageMatrix = getImageMatrix();
            // Invert it to map view coordinates back to bitmap coordinates
            currentImageMatrix.invert(viewToBitmapMatrix);
            imageMatrixCalculated = true;
        } else {
            imageMatrixCalculated = false; // Cannot calculate yet
        }
    }

    // Called when the view's layout changes (including initial layout)
    @Override
    public void onLayoutChange(View v, int left, int top, int right, int bottom,
                               int oldLeft, int oldTop, int oldRight, int oldBottom) {
        // The image matrix might change when layout changes, so update it.
        updateMatrixIfNeeded();
    }

    /**
     * Retrieves the drawn mask as a single-channel Bitmap (ALPHA_8).
     * The dimensions of the returned bitmap will match the original input bitmap.
     * Pixels in the returned bitmap represent the alpha/opacity of the mask
     * (derived from the ARGB maskBitmap).
     *
     * @return A Bitmap with Bitmap.Config.ALPHA_8 containing the mask,
     *         or null if no mask has been created or if the original bitmap is null.
     */
    @Nullable
    public Bitmap getMask() {
        if (maskBitmap == null || maskBitmap.isRecycled()) {
            return null;
        }

        // Create a new bitmap configured to store only alpha values
        // Its dimensions match the original bitmap.
        Bitmap alphaMask = null;
        try {
            alphaMask = Bitmap.createBitmap(maskBitmap.getWidth(), maskBitmap.getHeight(), Bitmap.Config.ALPHA_8);

            // Efficiently extract alpha channel.
            // Since we draw solid color (like RED) onto an ARGB_8888 bitmap,
            // the alpha channel directly represents the mask coverage.
            int[] pixels = new int[maskBitmap.getWidth() * maskBitmap.getHeight()];
            maskBitmap.getPixels(pixels, 0, maskBitmap.getWidth(), 0, 0, maskBitmap.getWidth(), maskBitmap.getHeight());

            // Allocate buffer for ALPHA_8 (1 byte per pixel)
            ByteBuffer alphaByteBuffer = ByteBuffer.allocate(pixels.length);
            for (int pixel : pixels) {
                // Extract the alpha component (most significant byte).
                // If the pixel was touched (e.g., solid red ARGB = 0xFFFF0000), alpha is FF.
                // If transparent (ARGB = 0x00000000), alpha is 00.
                if(((pixel >> 24) & 0xFF)<16){
                    alphaByteBuffer.put((byte) 0xFF); // Set alpha to FF (opaque)
                }else{
                    alphaByteBuffer.put((byte) 0x00); // Set alpha to 00 (transparent)
                }

//                if(pixel>128)
//                    alphaByteBuffer.put((byte) 0xFF); // Set alpha to FF (opaque)
//                else
//                    alphaByteBuffer.put((byte) 0x00); // Set alpha to 00 (transparent)
//                alphaByteBuffer.put((byte) ((pixel >> 24) & 0xFF));
            }

            alphaByteBuffer.rewind(); // Reset buffer position before reading
            alphaMask.copyPixelsFromBuffer(alphaByteBuffer); // Copy alpha data into the ALPHA_8 bitmap

        } catch (OutOfMemoryError e) {
            System.err.println("Failed to create alpha mask bitmap: " + e.getMessage());
            return null; // Return null if allocation fails
        } catch (Exception e) { // Catch other potential issues
            System.err.println("Failed getMask: " + e.getMessage());
            if (alphaMask != null && !alphaMask.isRecycled()) alphaMask.recycle(); // Clean up if partially created
            return null;
        }

        return alphaMask;
    }

    /**
     * Gets the current mask drawing paint object to allow customization
     * (e.g., changing color, stroke width). Remember to call invalidate()
     * after changing paint properties if you want the view to reflect changes
     * for future drawing.
     *
     * @return The Paint object used for drawing mask strokes.
     */
    @NonNull
    public Paint getMaskDrawingPaint() {
        return maskDrawingPaint;
    }

    /**
     * Gets the current mask overlay paint object to allow customization
     * (e.g., changing alpha). Remember to call invalidate() after changing.
     *
     * @return The Paint object used for drawing the mask overlay.
     */
    @NonNull
    public Paint getMaskOverlayPaint() {
        return maskOverlayPaint;
    }


    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        // Clean up resources when the view is detached
        recycleBitmaps();
        removeOnLayoutChangeListener(this); // Remove listener
    }

    // Optional: Consider adding methods to recycle bitmaps if needed for specific lifecycle management outside detachment
    public void recycleBitmaps() {
        // Be careful recycling originalBitmap if it might be used elsewhere
        // if (originalBitmap != null && !originalBitmap.isRecycled()) {
        //     originalBitmap.recycle();
        //     originalBitmap = null;
        // }
        clearMaskInternal(); // Recycles maskBitmap
    }
}