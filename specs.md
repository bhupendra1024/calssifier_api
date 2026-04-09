# Context Overlap Analyzer — System Specification
**Version:** v1.0  
**Deployment:** Local-first  
**Primary Objective:** Detect semantic overlap between classification contexts using CLIP embeddings for text-to-image verification workflows.

---

## 1. Overview

The system is designed to analyze semantic similarity between multiple text-based classification contexts by embedding them into a shared CLIP vector space.

Its purpose is to identify:

- overlapping classifications
- semantically ambiguous labels
- duplicate or near-duplicate context definitions
- classification collision risks during image verification

The system must run fully locally and support iterative manual refinement of context descriptions.

---

## 2. Problem Statement

Given a set of classification descriptions such as:

- Solar Panel with Farmer
- Foundation with Farmer
- Controller with Farmer
- Water Discharge with Farmer
- Pump Serial Number
- Motor Serial Number

the system must determine how semantically close each context is to every other context.

This helps identify:

- classes that overlap excessively
- classes likely to produce false-positive image matches
- contexts requiring prompt refinement

---

## 3. Core Functional Requirements

### 3.1 Context Ingestion
The system shall accept classification contexts in the form:

```json
{
  "Solar Panel with Farmer": "A clear image of a farmer near or interacting with a solar panel installation, where both the farmer and the solar panel are prominently visible and occupy a significant portion of the frame. The farmer should be easily identifiable as a human subject, and the solar panel structure should be clearly recognizable. Other elements like trees, land, or background objects may be present but should not dominate the image.",
  "Foundation with Farmer": "A clear image of a farmer near or interacting with a control panel installed on a cemented foundation, where the farmer, the control panel, and the cemented base are all prominently visible and occupy a significant portion of the frame. All three elements should be clearly identifiable, with the farmer as a human subject, the control panel as a device or box, and the cemented foundation as a solid base structure. Background elements may be present but should not dominate the image.",
  "Controller with Farmer": "A clear image of a farmer near or interacting with a control panel, where both the farmer and the controller are prominently visible and occupy a significant portion of the frame. The farmer should be clearly identifiable as a human subject, and the control panel should be clearly recognizable as a device or box. Both must be present in the same image, with background elements being secondary.",
  "Front side Panel module": "A clear front-facing view of a solar panel module, where the panel surface is fully visible from the front side and occupies a significant portion of the frame. The grid pattern of the panel cells should be clearly visible, with minimal obstruction. The panel should be the main subject, with background elements like land or sky being secondary.",
  "Water Discharge with Farmer": "A clear image of a farmer standing near or interacting with a water discharge source, where flowing water is visibly coming out of a pipe or pump. Both the farmer and the water discharge must be prominently visible and occupy a significant portion of the frame. The water flow should be clearly identifiable, and the farmer should be clearly visible as a human subject. Background elements should be secondary.",
  "Inside IMEI Code": "A clear image showing the inside of a device with an IMEI code printed on a label or sticker. The IMEI number should be clearly visible and readable, typically located inside a battery compartment or inner panel. The label containing the IMEI should occupy a significant portion of the image, with the text clearly legible. Background elements should be minimal and not distracting.",
  "Adhar Card": "A clear image of an Aadhaar card, where the card is prominently visible and occupies a significant portion of the frame. The card should show identifiable features such as the Aadhaar layout, government emblem, photo and text fields. The document should be clearly recognizable as an Aadhaar card, with minimal background distraction.",
  "Pump Serial Number": "A clear image showing a pump with a serial number printed on a label, sticker, or metal plate. The serial number should be clearly visible and readable, typically located on the pump body or attached plate. The label containing the serial number should occupy a significant portion of the frame, with the text clearly legible. The pump or machine context should also be visible.",
  "Motor Serial Number": "A clear image showing an electric motor with a serial number printed on a label, sticker, or metal nameplate. The serial number should be clearly visible and readable, typically located on the motor body or attached plate. The label containing the serial number should occupy a significant portion of the frame, with the text clearly legible. The motor should also be visible to establish context.",
  "Controller Serial Number": "A clear image showing a controller or control panel with a serial number printed on a label, sticker, or metal plate. The serial number should be clearly visible and readable, typically located on the controller box or panel surface. The label containing the serial number should occupy a significant portion of the frame, with the text clearly legible. The controller device should also be visible to establish context.",
  "Water filling with civit mat": "A clear image showing water being discharged or filling onto a cemented or concrete platform, where the flowing water is visibly coming out of a pipe or outlet. The cemented base should be clearly visible and identifiable, and the water flow should be prominent. Both the water discharge and the concrete platform should occupy a significant portion of the frame, with background elements being secondary.",
  "Earthing and Lighting": "A clear image showing an electrical earthing system along with lightning protection components. The image should include visible grounding elements such as earthing wires, rods, or grounding strips connected to equipment, and a lightning arrester or conductor system. Both the earthing setup and lightning protection elements should be clearly identifiable and occupy a significant portion of the frame, with background elements being secondary.",
  "Outside contour": "A clear image showing the outside contour of an installation or system, capturing the full external view and overall boundary. The entire structure should be visible within the frame, showing its shape, layout, and surrounding context. The image should not focus on specific components but instead provide a complete exterior view of the setup. Background elements like land or sky may be present but should not obscure the structure."
}
 

Result: Give the results of all the classifer so then it can be used to plot a graph

cx * cx in a ui, also need to store the embeddings in postgres db 

