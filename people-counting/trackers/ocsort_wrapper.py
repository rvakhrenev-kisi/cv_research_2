"""
OC-SORT wrapper for integration with our people counting system.
This provides a compatibility layer between OC-SORT and our existing tracking interface.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from .ocsort.ocsort import OCSort as OCSortCore


class OCSortWrapper:
    """
    Wrapper for OC-SORT tracker that provides compatibility with our existing system.
    """
    
    def __init__(self, 
                 det_thresh: float = 0.12,
                 max_age: int = 40,
                 min_hits: int = 2,
                 iou_threshold: float = 0.25,
                 delta_t: int = 3,
                 asso_func: str = "iou",
                 inertia: float = 0.4,
                 use_byte: bool = False):
        """
        Initialize OC-SORT tracker with parameters.
        
        Args:
            det_thresh: Detection confidence threshold
            max_age: Maximum age of tracks before deletion
            min_hits: Minimum hits before confirming a track
            iou_threshold: IoU threshold for association
            delta_t: Temporal window for observation
            asso_func: Association function type
            inertia: Motion inertia parameter
            use_byte: Whether to use BYTE association
        """
        self.ocsort = OCSortCore(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=delta_t,
            asso_func=asso_func,
            inertia=inertia,
            use_byte=use_byte
        )
        self.frame_count = 0
        
    def update_with_detections(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: Supervision Detections object
            
        Returns:
            Updated detections with tracking IDs
        """
        try:
            if detections.xyxy is None or len(detections.xyxy) == 0:
                # No detections, update with empty array
                output_results = np.empty((0, 5))
                img_info = (480, 640)  # Default image size
                img_size = (640, 640)  # Default input size
            else:
                # Convert detections to OC-SORT format
                scores = detections.confidence if detections.confidence is not None else np.ones(len(detections.xyxy))
                output_results = np.column_stack([detections.xyxy, scores])
                img_info = (480, 640)  # We'll need to get actual image size
                img_size = (640, 640)  # Default input size
            
            # Update OC-SORT tracker
            tracked_results = self.ocsort.update(output_results, img_info, img_size)
            
            if len(tracked_results) == 0:
                # No tracks, return empty detections
                return detections
            
            # Convert back to supervision format
            if len(tracked_results.shape) == 1:
                tracked_results = tracked_results.reshape(1, -1)
            
            # Extract bounding boxes and IDs
            if tracked_results.shape[1] >= 5:
                xyxy = tracked_results[:, :4]
                track_ids = tracked_results[:, 4].astype(int)
                
                # Create new detections object with tracking IDs
                import supervision as sv
                new_detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=detections.confidence[:len(xyxy)] if detections.confidence is not None else None,
                    class_id=detections.class_id[:len(xyxy)] if detections.class_id is not None else None,
                    tracker_id=track_ids
                )
                return new_detections
            
            return detections
        except Exception as e:
            # If there's an error, return the original detections
            print(f"OC-SORT error: {e}")
            return detections
    
    def reset(self):
        """Reset the tracker state."""
        self.ocsort = OCSortCore(
            det_thresh=self.ocsort.det_thresh,
            max_age=self.ocsort.max_age,
            min_hits=self.ocsort.min_hits,
            iou_threshold=self.ocsort.iou_threshold,
            delta_t=self.ocsort.delta_t,
            asso_func="iou",  # Default association function
            inertia=self.ocsort.inertia,
            use_byte=self.ocsort.use_byte
        )
        self.frame_count = 0
