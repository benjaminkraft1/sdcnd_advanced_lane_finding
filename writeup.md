## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

please see the notebook for the real output of this project

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/P2_advanced_lane_finding.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

[image7]:  ./output_images/Distortedimage-Undistortedimage.png

### Pipeline 

The pipeline can be used for single images and video frames.
`lane_finding = LaneFinding()`

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
(Please see output in the junypter notebook)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Here's an example of my output for this step.  (note: this is not actually from one of the test images)

(Please see output in the junypter notebook)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    # Src Image quadrangle verties coordinates 
    src = np.float32([[280,  700],  # Bottom left
                      [595,  460],  # Top left
                      [725,  460],  # Top right
                      [1125, 700]]) # Bottom right
    
    # Dst Image quadrangle verties coordinates
    dst = np.float32([[250,  720],  # Bottom left
                      [250,    0],  # Top left
                      [1065,   0],  # Top right
                      [1065, 720]]) # Bottom right 
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

(Please see output in the junypter notebook)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

(Please see output in the junypter notebook)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in my code in `measure_curvature()` and in `car_offset()`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

(Please see output in the junypter notebook)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_solution.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My Pipeline goes trough the following steps:

* Initialization: In the Init Phase the Camera Object is calibrated and the image transformation is setup to use HLS Color Space

##### Loop
* Undistort image using `self.camera(img)`
* Apply thresholding functions `self.transform.combined_binary(undistorted_img)`
* Transform perspective over the combined warped image `perspective_transform(combined_binary)`
* Get lane lines `search_around_poly(combined_warped_image, self.lanes_fit, True)` << this method switched back to sliding windows if there are no lanes found 
* Calculate the radius of curvature in pixels for both lane lines `measure_curvature(left_points[1], self.lanes_fit[0], self.lanes_fit[1])`
* Calculate Position of the car relative to the lane center `car_offset(leftx=left_points[0], rightx=right_points[0], img_shape=combined_warped_image.shape)`
* Overlay lane information       


#### Issues
The pipeline had some issues to detect lanes correctly when there were shadows on the street. I'm now using the HLS S channel for gradient thresholds and the HLS L Channel for the color channel threshold, since this fits better.
In total it is now a little unprecise close to the car on bright road situations, but this is still a much better result than before.