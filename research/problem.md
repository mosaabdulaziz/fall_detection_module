# Problem Frame: Pose-Based Fall, Lying-Person, and Prolonged Immobility Detection Pipeline

## 1. Task & User Story
As a health and safety manager, the system must identify:
1. **Potential falls** (candidate fall events from video sequences showing fall-like events).
2. **Persons lying on the floor** (candidate lying-person events when lying posture persists for a configured duration).
3. **Prolonged unusual immobility** (candidate immobility events when a tracked person exhibits little observable movement over a configured duration).

All detections are **candidate events** brought to an operator's attention (distinguishing candidate detections from confirmed incidents).

## 2. Problem Type & Confirmed Research Mode
- **Problem Type**: Architecture choice & adaptation under constraints on top of frozen `yolo26x-pose.pt` + `ByteTrack`.
- **Confirmed Depth Mode**: `standard` (8–15 sources investigated + GitHub code & weight verification).

## 3. Confirmed Architecture Scope (Updated per User Round 0 & User Round 1)
- **Conditional Rule from User Round 0 (`A1` at Phase 1 Exit)**:
  - *"if there is a codebase and weights already present for priority 1 then lets go with option 2 [Priority 1 Temporal Models], if there is no open weights and we will require training or finetuning, or there is no codebases found for priority 1 then we will go with option 3 [Priority 2 Kinematic Rules]"*.
- **Confirmed Selection in User Round 1 (`A1` & `A2` at Phase 4 Exit)**:
  - Because `GajuuzZ/Human-Falling-Detect-Tracks` (`S12`) provides a complete codebase and downloadable pretrained weights (`tsstg-model.pth`) trained on Le2i for 7 classes (`Standing`, `Walking`, `Sitting`, `Lying Down`, `Stand up`, `Sit down`, `Fall Down`), the user selected **Pure Priority 1**: center the design directly around integrating the pretrained `TSSTG` (`tsstg-model.pth`) temporal model on top of `yolo26x-pose.pt` + `ByteTrack`, using lightweight rules only where necessary for duration/immobility tracking and fallback guards.

## 4. Confirmed Evaluation & Dataset Scope
- **Primary Evaluation**: **Le2i Fall Detection Dataset** + custom/recorded CCTV test sequences.
- **Secondary Augmentation**: Public multi-class fall/ADL benchmarks (e.g., UR Fall, UP-Fall).

## 5. Inputs and Outputs
- **Inputs**: Recorded video sequences (`.mp4`, `.avi`), webcam, or RTSP streams (`fps` ~25-30).
- **Outputs**: Annotated video + JSONL stream of candidate events (`type`: `fall` | `lying` | `immobility`, `track`, `t_start`, `t_candidate`, `t_alert`, `priority`, and event evidence/metrics) for operator review.
