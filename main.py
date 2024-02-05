import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def get_landmarks(image):
    results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0].landmark
    landmarks = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in face_landmarks]
    return landmarks

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warp_triangles(img1, img2, tri1, tri2):
    r1 = cv2.boundingRect(np.float32([tri1]))
    r2 = cv2.boundingRect(np.float32([tri2]))
    tri1_rect = []
    tri2_rect = []
    tri2_rect_int = []

    for i in range(0, 3):
        tri1_rect.append(((tri1[i][0] - r1[0]), (tri1[i][1] - r1[1])))
        tri2_rect.append(((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))
        tri2_rect_int.append(
            (int(tri2[i][0] - r2[0]), int(tri2[i][1] - r2[1])))

    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    img2_rect = np.zeros((r2[3], r2[2], 3), dtype=img1_rect.dtype)
    img2_rect = apply_affine_transform(
        img1_rect, tri1_rect, tri2_rect, (r2[2], r2[3]))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2_rect_int), (1.0, 1.0, 1.0), 16, 0)
    img2_rect = img2_rect * mask
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1.0 - mask) + img2_rect

def get_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    triangle_list = subdiv.getTriangleList()
    delaunay_tri = []

    for t in triangle_list:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        # Check if points are within the bounding rectangle
        if pt1[0] >= rect[0] and pt1[1] >= rect[1] and pt1[0] < rect[0] + rect[2] and pt1[1] < rect[1] + rect[3] and \
           pt2[0] >= rect[0] and pt2[1] >= rect[1] and pt2[0] < rect[0] + rect[2] and pt2[1] < rect[1] + rect[3] and \
           pt3[0] >= rect[0] and pt3[1] >= rect[1] and pt3[0] < rect[0] + rect[2] and pt3[1] < rect[1] + rect[3]:
            
            ind = []
            for p in [pt1, pt2, pt3]:
                for k, point in enumerate(points):
                    if abs(p[0] - point[0]) < 1.0 and abs(p[1] - point[1]) < 1.0:
                        ind.append(k)
                        break
            if len(ind) == 3:
                delaunay_tri.append((ind[0], ind[1], ind[2]))

    return delaunay_tri

def match_histograms(source, reference):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    """
    # Split the images into their respective channels
    source_channels = cv2.split(source)
    reference_channels = cv2.split(reference)

    matched_channels = []
    for s_channel, r_channel in zip(source_channels, reference_channels):
        # Calculate the histograms of the source and reference images
        s_hist, _ = np.histogram(s_channel.flatten(), 256, [0, 256])
        r_hist, _ = np.histogram(r_channel.flatten(), 256, [0, 256])

        # Calculate the cumulative distribution function of the histograms
        s_cdf = np.cumsum(s_hist) / float(np.sum(s_hist))
        r_cdf = np.cumsum(r_hist) / float(np.sum(r_hist))

        # Create a lookup table to map pixel values from source to reference
        lookup_table = np.zeros(256)
        g = 0
        for i in range(256):
            while r_cdf[g] < s_cdf[i] and g < 255:
                g += 1
            lookup_table[i] = g

        # Apply the mapping to get the channel of the matched image
        matched_channel = cv2.LUT(np.uint8(s_channel), np.uint8(lookup_table))
        matched_channels.append(matched_channel)

    # Merge the matched channels back together
    matched_image = cv2.merge(matched_channels)
    return matched_image

def process_video(static_image_path, video_path, output_video_path):
    print("Loading static image...")
    static_img = cv2.imread(static_image_path)
    static_landmarks = get_landmarks(static_img)
    if static_landmarks is None:
        print("Error in detecting landmarks in static image.")
        return

    static_rect = (0, 0, static_img.shape[1], static_img.shape[0])
    static_delaunay_tri = get_delaunay_triangles(static_rect, static_landmarks)

    print("Opening video file...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error reading video file.")
        cap.release()
        return

    height, width, layers = first_frame.shape
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_landmarks = get_landmarks(frame)
        if frame_landmarks is None:
            print("Landmarks not detected in frame, skipping...")
            continue

        warped_frame = np.copy(frame)
        for tri_indices in static_delaunay_tri:
            tri1 = [static_landmarks[tri_indices[0]], static_landmarks[tri_indices[1]], static_landmarks[tri_indices[2]]]
            tri2 = [frame_landmarks[tri_indices[0]], frame_landmarks[tri_indices[1]], frame_landmarks[tri_indices[2]]]
            warp_triangles(static_img, warped_frame, tri1, tri2)

        # Apply color correction
        corrected_warped_frame = match_histograms(warped_frame, frame)

        video_writer.write(corrected_warped_frame)

    cap.release()
    video_writer.release()
    print("Video processing complete.")

if __name__ == "__main__":
    static_image_path = 'task_2/sardor.jpg'
    video_path = 'task_2/sardoriy.mp4'
    output_video_path = 'task_2/output_demo.mp4'
    process_video(static_image_path, video_path, output_video_path)