package com.quicinc;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.widget.ImageView;

public class MaskDrawingView extends androidx.appcompat.widget.AppCompatImageView {
    private Bitmap mOriginalBitmap;
    private Bitmap mMaskBitmap;
    private Canvas mMaskCanvas;
    private Paint mPaint;
    private Path mPath;
    private boolean mShowMask = false;

    public MaskDrawingView(Context context) {
        super(context);
        init();
    }

    public MaskDrawingView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        mPaint = new Paint();
        mPaint.setColor(Color.RED); // 遮罩颜色
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeWidth(20f); // 画笔宽度
        mPaint.setStrokeCap(Paint.Cap.ROUND);
        mPaint.setStrokeJoin(Paint.Join.ROUND);
        mPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.SRC_OVER));

        mPath = new Path();
    }

    public void setImageResource(int resId) {
        setImage(BitmapFactory.decodeResource(getResources(), resId), true);
    }

    public void setMask(boolean showMask) {
        this.mShowMask = showMask;

        mMaskBitmap = Bitmap.createBitmap(mOriginalBitmap.getWidth(), mOriginalBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        mMaskCanvas = new Canvas(mMaskBitmap);
        mMaskCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
    }

    public void setImage(Bitmap bitmap) {
        setImage(bitmap, false);
    }

    public void setImage(Bitmap bitmap, boolean showMask) {
        this.mOriginalBitmap = bitmap;
        this.mShowMask = showMask;


        if (bitmap != null) {
            Log.w("setImage","showMask="+showMask);
            mMaskBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
            mMaskCanvas = new Canvas(mMaskBitmap);
            mMaskCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
        } else {
            Log.w("setImage","showMask="+showMask+", bitmap==null");
            mMaskBitmap = null;
            mMaskCanvas = null;
        }


        setImageBitmap(mOriginalBitmap);
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (mShowMask && mMaskBitmap != null) {
            // 绘制遮罩层
            canvas.drawBitmap(mMaskBitmap, 0, 0, null);
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (!mShowMask || mOriginalBitmap == null) {
            return super.onTouchEvent(event);
        }

        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                mPath.moveTo(x, y);
                break;
            case MotionEvent.ACTION_MOVE:
                mPath.lineTo(x, y);
                mMaskCanvas.drawPath(mPath, mPaint);
                invalidate();
                break;
            case MotionEvent.ACTION_UP:
                mPath.reset();
                break;
        }

        return true;
    }

    public Bitmap getMaskedBitmap() {
        if (mOriginalBitmap == null) return null;

        Bitmap result = mOriginalBitmap.copy(Bitmap.Config.ALPHA_8, true);
        Canvas canvas = new Canvas(result);

        if (mMaskBitmap != null) {
            canvas.drawBitmap(mMaskBitmap, 0, 0, null);
        }

        return result;
    }
}