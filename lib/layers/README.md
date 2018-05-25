## faster rcnn layers

#### generate_anchors
generate and shift anchors with default scales (8, 16, 32) and default ratios (0.5, 1, 2) on feature maps  

#### anchor_target
generate training targets/labels for each anchor (label 1 is foreground, label 0 is background, label -1 is ignored). regression (smoothl1) on foreground anchors

#### proposal
convert anchors (on rpn) to proposal (on fast rcnn)

#### proposal_target
generate training targets/labels for each proposal (label 0 is background, label 1-K is objects' label). regression (smoothl1) 
on objects' proposal (label > 0)

#### detection
generate detection from proposal on fast rcnn