from ultralytics import YOLO

class YOLOPoseDetector:
    """
    YOLO Pose人体关键点检测
    """

    def __init__(
        self,
        model_path = "yolo11n-pose.pt"
    ):
        self.model = YOLO(model_path)

    def predict(
        self,
        image_path,
        conf = 0.3
    ):
        results = self.model(
            image_path,
            conf = conf
        )

        persons = []

        for result in results:

            boxes = result.boxes

            keypoints = result.keypoints

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                score = (
                    box.conf[0]
                    .cpu()
                    .item()
                )

                kp = None

                if keypoints is not None:
                    kp = (
                        keypoints.xy[i]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                persons.append({
                    "label": "perosn",
                    "score": round(score, 4),
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ],
                    "keypoints": kp,
                    "source": "YOLO-Pose"
                })

        return persons