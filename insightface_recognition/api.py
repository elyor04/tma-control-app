"""
A module that replicates a subset of the face_recognition API using InsightFace.
Inspired by the official InsightFace FaceAnalysis class and examples, this module
provides convenient wrapper functions for face detection, landmark localization,
and embedding extraction.

Provided functions:
  • set_default_app()         – (Re)initializes the global FaceAnalysis instance.
  • get_default_app()         – Returns the global FaceAnalysis instance.
  • get_app()                 – Returns a new FaceAnalysis instance.
  • get_faces()               – Detects faces in an image using the default FaceAnalysis instance.
  • load_image_file()         – Loads an image file in BGR format using OpenCV.
  • face_locations()          – Returns bounding boxes (top, right, bottom, left) for detected faces.
  • face_landmarks()          – Returns 5-point facial landmarks for each detected face.
  • face_encodings()          – Extracts face embeddings for each detected face.
  • face_distance()           – Computes a distance (1 - cosine similarity) between embeddings.
  • face_similarity()         – Computes cosine similarity between embeddings.
  • compare_faces()           – Returns a list of booleans indicating matches based on a tolerance.
  • compare_faces_v2()        – Returns the best match name, best similarity score, and a match flag.

Note: InsightFace’s FaceAnalysis and underlying models expect images in BGR format,
thus load_image_file returns images in BGR, matching the official demo.
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Global singleton for the FaceAnalysis instance.
_default_app = None


def set_default_app(
    model_name="buffalo_l",
    root="./.insightface",
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"],
    ctx_id=-1,
    det_thresh=0.5,
    det_size=(640, 640),
):
    """
    Initializes the global FaceAnalysis instance.

    Parameters:
      model_name (str): Name of the InsightFace model (default "buffalo_l").
      root (str): Directory for model cache/downloads.
      allowed_modules (list or None): Optional list to restrict module loading
                                      (e.g., ["detection", "recognition"]).
      providers (list): Execution providers (e.g., ["CPUExecutionProvider"]).
      ctx_id (int): Context id (-1 for CPU, 0 for GPU).
      det_thresh (float): Face detection threshold (default 0.5).
      det_size (tuple): Input size for face detection (default (640, 640)).
    """
    global _default_app

    _default_app = FaceAnalysis(
        name=model_name,
        root=root,
        allowed_modules=allowed_modules,
        providers=providers,
    )
    _default_app.prepare(
        ctx_id=ctx_id,
        det_thresh=det_thresh,
        det_size=det_size,
    )


def get_default_app():
    """
    Returns the global FaceAnalysis instance, initializing it if necessary.
    """
    if _default_app is None:
        set_default_app()

    return _default_app


def get_app(
    model_name="buffalo_l",
    root="./.insightface",
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"],
    ctx_id=-1,
    det_thresh=0.5,
    det_size=(640, 640),
):
    """
    Get FaceAnalysis instance.

    Parameters:
      model_name (str): Name of the InsightFace model (default "buffalo_l").
      root (str): Directory for model cache/downloads.
      allowed_modules (list or None): Optional list to restrict module loading
                                      (e.g., ["detection", "recognition"]).
      providers (list): Execution providers (e.g., ["CPUExecutionProvider"]).
      ctx_id (int): Context id (-1 for CPU, 0 for GPU).
      det_thresh (float): Face detection threshold (default 0.5).
      det_size (tuple): Input size for face detection (default (640, 640)).
    """
    app = FaceAnalysis(
        name=model_name,
        root=root,
        allowed_modules=allowed_modules,
        providers=providers,
    )
    app.prepare(
        ctx_id=ctx_id,
        det_thresh=det_thresh,
        det_size=det_size,
    )
    return app


def get_faces(image, minSize=(50, 50), maxSize=(5000, 5000), maxNum=5, reverse=True, *, app=None):
    """
    Detects faces in the input image and then filters them by size.

    Parameters:
      image (np.ndarray): Input image in BGR format.
      minSize (tuple): Minimum size (height, width) for a face bounding box.
      maxSize (tuple): Maximum size (height, width) for a face bounding box.
      maxNum (int): Maximum number of Face objects to return.
      reverse (bool): If True, sort faces in descending order by area (largest first).
      app (FaceAnalysis): Optional FaceAnalysis instance. If None, uses the default instance.

    Returns:
      list: A list of valid Face objects.
    """
    if app is None:
        app = get_default_app()

    faces = app.get(image, max_num=maxNum + 2)

    h, w, _ = image.shape
    valid_faces = []

    for face in faces:
        # FaceAnalysis returns bounding box as [left, top, right, bottom]
        bbox = face.bbox.astype(int)

        # Clip bounding box to image boundaries.
        left = max(0, bbox[0])
        top = max(0, bbox[1])
        right = min(w, bbox[2])
        bottom = min(h, bbox[3])

        # Calculate height and width.
        height = bottom - top
        width = right - left

        # Accept only if the box is non-zero and meets the size constraints.
        if (top < bottom and left < right) and (minSize[0] <= height <= maxSize[0] and minSize[1] <= width <= maxSize[1]):
            valid_faces.append(face)

    # Sort valid faces by area.
    valid_faces.sort(key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=reverse)

    return valid_faces[:maxNum]


def load_image_file(file):
    """
    Loads an image file into a NumPy array using OpenCV.

    Note: The returned image is in BGR order, matching the expectation of InsightFace.

    Parameters:
      file (str): Path to the image file.

    Returns:
      np.ndarray: The loaded image in BGR format.
    """
    image = cv2.imread(file)
    if image is None:
        raise IOError(f"Could not load image file: {file}")
    return image


def face_locations(image, *, faces=None):
    """
    Detects faces in an image and returns their bounding boxes in the format:
    (top, right, bottom, left).

    Internally, InsightFace's FaceAnalysis returns the bounding box as
    [left, top, right, bottom]. This function reorders the coordinates to match
    the face_recognition API standard.

    Parameters:
      image (np.ndarray): Input image in BGR format.
      faces (list or None): Pre-detected Face objects (optional). If None, detection is run.

    Returns:
      list of tuples: Each tuple is (top, right, bottom, left) for a detected face.
    """
    if faces is None:
        faces = get_faces(image)

    h, w, _ = image.shape
    locations = []

    for face in faces:
        bbox = face.bbox.astype(int)  # Original format: [left, top, right, bottom]

        # Clip the bounding box to the image boundaries.
        left = max(0, bbox[0])
        top = max(0, bbox[1])
        right = min(w, bbox[2])
        bottom = min(h, bbox[3])

        locations.append((top, right, bottom, left))

    return locations


def face_landmarks(image, *, faces=None):
    """
    Extracts five-point facial landmarks for each detected face.

    Each returned dictionary contains the following keys:
      "left_eye", "right_eye", "nose", "left_mouth", "right_mouth"
    representing the corresponding landmark coordinates.

    Parameters:
      image (np.ndarray): Input image in BGR format.
      faces (list or None): Pre-detected Face objects (optional). If None, detection is performed.

    Returns:
      list of dicts: Each dict holds landmark coordinates for one face.
    """
    if faces is None:
        faces = get_faces(image)

    landmarks_list = []
    for face in faces:
        landmark = {}
        if hasattr(face, "kps") and face.kps is not None and face.kps.shape[0] >= 5:
            kps = face.kps.astype(int)
            landmark["left_eye"] = (kps[0][0], kps[0][1])
            landmark["right_eye"] = (kps[1][0], kps[1][1])
            landmark["nose"] = (kps[2][0], kps[2][1])
            landmark["left_mouth"] = (kps[3][0], kps[3][1])
            landmark["right_mouth"] = (kps[4][0], kps[4][1])
        landmarks_list.append(landmark)
    return landmarks_list


def face_encodings(image, *, faces=None):
    """
    Extracts face embeddings for each detected face in the image.

    The embeddings are high-dimensional feature vectors used for face recognition tasks.

    Parameters:
      image (np.ndarray): Input image in BGR format.
      faces (list or None): Pre-detected Face objects (optional). If None, detection is performed.

    Returns:
      list of np.ndarray: A list of embedding vectors (one per detected face).
    """
    if faces is None:
        faces = get_faces(image)

    embeddings = []
    for face in faces:
        if hasattr(face, "embedding") and face.embedding is not None:
            embeddings.append(face.embedding)
    return embeddings


def face_distance(known_face_encodings, face_encoding, metric="euclidean"):
    """
    Computes a distance measure between known face embeddings and a candidate face embedding.

    Two metrics are supported:
      - "euclidean": Euclidean distance.
      - "cosine": 1 - cosine similarity, where lower means more similar.

    Parameters:
      known_face_encodings (list of np.ndarray): List of known face embeddings.
      face_encoding (np.ndarray): Candidate face embedding.
      metric (str): "euclidean" (default) or "cosine".

    Returns:
      np.ndarray: Array of distance values.
    """
    known = np.array(known_face_encodings)
    if metric == "euclidean":
        diff = known - face_encoding
        distances = np.linalg.norm(diff, axis=1)
    elif metric == "cosine":
        epsilon = 1e-8
        dot = np.sum(known * face_encoding, axis=1)
        norm_known = np.linalg.norm(known, axis=1) + epsilon
        norm_candidate = np.linalg.norm(face_encoding) + epsilon
        cosine_sim = dot / (norm_known * norm_candidate)
        distances = 1 - cosine_sim
    else:
        raise ValueError("Unsupported metric. Choose 'euclidean' or 'cosine'.")
    return distances


def face_similarity(known_face_encodings, face_encoding):
    """
    Computes the cosine similarity between a candidate face embedding and each of the known face embeddings.

    Cosine similarity is calculated as:
       similarity = dot(emb1, emb2) / (||emb1|| * ||emb2||)

    Parameters:
      known_face_encodings (list of np.ndarray): List of known face embeddings.
      face_encoding (np.ndarray): Candidate face embedding.

    Returns:
      np.ndarray: The cosine similarity values (one per known face).
    """
    similarities = []
    for known_face_encoding in known_face_encodings:
        similarity = np.dot(known_face_encoding, face_encoding) / (np.linalg.norm(known_face_encoding) * np.linalg.norm(face_encoding))
        similarities.append(similarity)
    return np.array(similarities)


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.35, metric="euclidean"):
    """
    Compares a candidate face embedding against a list of known face embeddings.

    A face is considered a match if its distance is less than or equal to the specified tolerance.

    Parameters:
      known_face_encodings (list of np.ndarray): Known face embeddings.
      face_encoding_to_check (np.ndarray): Candidate face embedding.
      tolerance (float): Distance threshold for a match.
      metric (str): "euclidean" (default) or "cosine".

    Returns:
      list of bool: For each known face, True if the distance is within tolerance, False otherwise.
    """
    distances = face_distance(known_face_encodings, face_encoding_to_check, metric=metric)
    return list(distances < tolerance)


def compare_faces_v2(known_faces, face_encoding, threshold=0.65):
    """
    Finds the best match from a dictionary of known faces (name to embeddings mapping)
    for the candidate face embedding using cosine similarity.

    Parameters:
      known_faces (dict): Dictionary where keys are names and values are lists of embeddings.
      face_encoding (np.ndarray): Candidate face embedding.
      threshold (float): Similarity threshold for a match.

    Returns:
      tuple: (best_match, best_score, is_match), where:
             - best_match (str): The name of the best matching face.
             - best_score (float): The highest cosine similarity score found.
             - is_match (bool): True if best_score > threshold, False otherwise.
    """
    best_match = None
    best_score = -1
    for name, embeddings in known_faces.items():
        for known_emb in embeddings:
            similarity = np.dot(face_encoding, known_emb) / (np.linalg.norm(face_encoding) * np.linalg.norm(known_emb))
            if similarity > best_score:
                best_score = similarity
                best_match = name
    return best_match, best_score, best_score > threshold
