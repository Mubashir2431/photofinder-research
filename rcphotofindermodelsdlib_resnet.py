warning: in the working copy of 'src/photofinder/models/dlib_resnet.py', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/src/photofinder/models/dlib_resnet.py b/src/photofinder/models/dlib_resnet.py[m
[1mindex 263f963..95ec916 100644[m
[1m--- a/src/photofinder/models/dlib_resnet.py[m
[1m+++ b/src/photofinder/models/dlib_resnet.py[m
[36m@@ -1,4 +1,4 @@[m
[31m-from __future__ import annotations[m
[32m+[m[32m﻿from __future__ import annotations[m
 import os[m
 import numpy as np[m
 from .base import FaceEmbedder, FaceEmbedding[m
[36m@@ -12,7 +12,7 @@[m [mclass DlibResnetEmbedder(FaceEmbedder):[m
         try:[m
             import dlib  # type: ignore[m
         except Exception as e:[m
[31m-            raise RuntimeError("dlib is not installed. `pip install -e '.[dlib]'`") from e[m
[32m+[m[32m            raise RuntimeError("dlib is not installed. Install dlib-bin or dlib.") from e[m
 [m
         shape_path = os.environ.get("DLIB_SHAPE_PREDICTOR_PATH")[m
         rec_path = os.environ.get("DLIB_FACE_REC_MODEL_PATH")[m
[36m@@ -28,13 +28,25 @@[m [mclass DlibResnetEmbedder(FaceEmbedder):[m
         self.face_rec = dlib.face_recognition_model_v1(rec_path)[m
 [m
     def embed(self, bgr_image: np.ndarray):[m
[31m-        # dlib expects RGB[m
[31m-        rgb = bgr_image[:, :, ::-1][m
[31m-        faces = self.detector(rgb)[m
[32m+[m[32m        # Convert BGR -> RGB and FORCE contiguous uint8 memory[m
[32m+[m[32m        rgb = np.ascontiguousarray(bgr_image[:, :, ::-1], dtype=np.uint8)[m
[32m+[m
[32m+[m[32m        # Upsample=1 improves detection on smaller faces[m
[32m+[m[32m        faces = self.detector(rgb, 1)[m
[32m+[m
         out = [][m
         for f in faces:[m
             shape = self.shape_predictor(rgb, f)[m
[32m+[m
[32m+[m[32m            # Now this call works because rgb is contiguous (no negative strides)[m
             v = self.face_rec.compute_face_descriptor(rgb, shape)[m
             emb = np.asarray(v, dtype=np.float32)[m
[31m-            out.append(FaceEmbedding(embedding=emb, bbox_xyxy=(f.left(), f.top(), f.right(), f.bottom())))[m
[32m+[m
[32m+[m[32m            out.append([m
[32m+[m[32m                FaceEmbedding([m
[32m+[m[32m                    embedding=emb,[m
[32m+[m[32m                    bbox_xyxy=(f.left(), f.top(), f.right(), f.bottom()),[m
[32m+[m[32m                )[m
[32m+[m[32m            )[m
[32m+[m
         return out[m
