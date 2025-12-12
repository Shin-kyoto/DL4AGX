import onnx

# Load the ONNX model
model_path = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/onnx_files/vadv1_prev.pts_bbox_head.forward/vadv1_prev.pts_bbox_head.forward.onnx"
model = onnx.load(model_path)

# Iterate through all nodes to find potential MSDA operations
msda_related_nodes = []
for node in model.graph.node:
    if any(keyword in node.name.lower() or keyword in node.op_type.lower() 
           for keyword in ["deform", "attention", "msdeform"]):
        msda_related_nodes.append((node.name, node.op_type))

print(f"Found {len(msda_related_nodes)} potential MSDA-related nodes:")
for name, op_type in msda_related_nodes:
    print(f"  - {name} (Operation type: {op_type})")
