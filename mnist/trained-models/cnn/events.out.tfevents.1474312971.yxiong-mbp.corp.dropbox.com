       �K"	  ����Abrain.Event:2��q#     �<&	`�����A"��
X
inputPlaceholder*
dtype0*
shape: *(
_output_shapes
:����������
]
PlaceholderPlaceholder*
dtype0*
shape: *'
_output_shapes
:���������

J
dropoutPlaceholder*
dtype0*
shape: *
_output_shapes
:
f
Reshape/shapeConst*
dtype0*%
valueB"����         *
_output_shapes
:
p
ReshapeReshapeinputReshape/shape*/
_output_shapes
:���������*
T0*
Tshape0
o
truncated_normal/shapeConst*
dtype0*%
valueB"             *
_output_shapes
:
Z
truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
\
truncated_normal/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *

seed *
T0*&
_output_shapes
: 
�
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
: 
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
: 
�
VariableVariable*
dtype0*
shape: *
	container *
shared_name *&
_output_shapes
: 
�
Variable/AssignAssignVariabletruncated_normal*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
q
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*&
_output_shapes
: 
R
ConstConst*
dtype0*
valueB *���=*
_output_shapes
: 
t

Variable_1Variable*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
Variable_1/AssignAssign
Variable_1Const*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
k
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes
: 
�
Conv2DConv2DReshapeVariable/read*/
_output_shapes
:��������� *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
]
addAddConv2DVariable_1/read*
T0*/
_output_shapes
:��������� 
K
ReluReluadd*
T0*/
_output_shapes
:��������� 
�
MaxPoolMaxPoolRelu*/
_output_shapes
:��������� *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
q
truncated_normal_1/shapeConst*
dtype0*%
valueB"          @   *
_output_shapes
:
\
truncated_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_1/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *

seed *
T0*&
_output_shapes
: @
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*&
_output_shapes
: @
{
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*&
_output_shapes
: @
�

Variable_2Variable*
dtype0*
shape: @*
	container *
shared_name *&
_output_shapes
: @
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
w
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*&
_output_shapes
: @
T
Const_1Const*
dtype0*
valueB@*���=*
_output_shapes
:@
t

Variable_3Variable*
dtype0*
shape:@*
	container *
shared_name *
_output_shapes
:@
�
Variable_3/AssignAssign
Variable_3Const_1*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
k
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes
:@
�
Conv2D_1Conv2DMaxPoolVariable_2/read*/
_output_shapes
:���������@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
a
add_1AddConv2D_1Variable_3/read*
T0*/
_output_shapes
:���������@
O
Relu_1Reluadd_1*
T0*/
_output_shapes
:���������@
�
	MaxPool_1MaxPoolRelu_1*/
_output_shapes
:���������@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
i
truncated_normal_2/shapeConst*
dtype0*
valueB"@     *
_output_shapes
:
\
truncated_normal_2/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_2/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:
��
�
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0* 
_output_shapes
:
��
u
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0* 
_output_shapes
:
��
�

Variable_4Variable*
dtype0*
shape:
��*
	container *
shared_name * 
_output_shapes
:
��
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
q
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��
V
Const_2Const*
dtype0*
valueB�*���=*
_output_shapes	
:�
v

Variable_5Variable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Variable_5/AssignAssign
Variable_5Const_2*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�
`
Reshape_1/shapeConst*
dtype0*
valueB"����@  *
_output_shapes
:
q
	Reshape_1Reshape	MaxPool_1Reshape_1/shape*(
_output_shapes
:����������*
T0*
Tshape0
�
MatMulMatMul	Reshape_1Variable_4/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:����������
X
add_2AddMatMulVariable_5/read*
T0*(
_output_shapes
:����������
H
Relu_2Reluadd_2*
T0*(
_output_shapes
:����������
U
dropout_1/ShapeShapeRelu_2*
out_type0*
T0*
_output_shapes
:
a
dropout_1/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
a
dropout_1/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape*
dtype0*
seed2 *

seed *
T0*0
_output_shapes
:������������������
�
dropout_1/random_uniform/subSubdropout_1/random_uniform/maxdropout_1/random_uniform/min*
T0*
_output_shapes
: 
�
dropout_1/random_uniform/mulMul&dropout_1/random_uniform/RandomUniformdropout_1/random_uniform/sub*
T0*0
_output_shapes
:������������������
�
dropout_1/random_uniformAdddropout_1/random_uniform/muldropout_1/random_uniform/min*
T0*0
_output_shapes
:������������������
Z
dropout_1/addAdddropoutdropout_1/random_uniform*
T0*
_output_shapes
:
J
dropout_1/FloorFloordropout_1/add*
T0*
_output_shapes
:
H
dropout_1/DivDivRelu_2dropout*
T0*
_output_shapes
:
g
dropout_1/mulMuldropout_1/Divdropout_1/Floor*
T0*(
_output_shapes
:����������
i
truncated_normal_3/shapeConst*
dtype0*
valueB"   
   *
_output_shapes
:
\
truncated_normal_3/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
^
truncated_normal_3/stddevConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes
:	�

�
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:	�

t
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:	�

~

Variable_6Variable*
dtype0*
shape:	�
*
	container *
shared_name *
_output_shapes
:	�

�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�

p
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�

T
Const_3Const*
dtype0*
valueB
*���=*
_output_shapes
:

t

Variable_7Variable*
dtype0*
shape:
*
	container *
shared_name *
_output_shapes
:

�
Variable_7/AssignAssign
Variable_7Const_3*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:

k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:

�
MatMul_1MatMuldropout_1/mulVariable_6/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������

Y
add_3AddMatMul_1Variable_7/read*
T0*'
_output_shapes
:���������

J
outputSoftmaxadd_3*
T0*'
_output_shapes
:���������

D
LogLogoutput*
T0*'
_output_shapes
:���������

N
mulMulPlaceholderLog*
T0*'
_output_shapes
:���������

d
xentropy/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
{
xentropySummulxentropy/reduction_indices*#
_output_shapes
:���������*
T0*
	keep_dims( *

Tidx0
B
NegNegxentropy*
T0*#
_output_shapes
:���������
Q
Const_4Const*
dtype0*
valueB: *
_output_shapes
:
a
xentropy_meanMeanNegConst_4*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
d
ArgMaxArgMaxoutputArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
m
ArgMax_1ArgMaxPlaceholderArgMax_1/dimension*#
_output_shapes
:���������*
T0*

Tidx0
Q
accuracyEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
S
CastCastaccuracy*

DstT0*

SrcT0
*#
_output_shapes
:���������
Q
Const_5Const*
dtype0*
valueB: *
_output_shapes
:
b
accuracy_meanMeanCastConst_5*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
`
ScalarSummary/tagsConst*
dtype0*
valueB Bxentropy_mean*
_output_shapes
: 
b
ScalarSummaryScalarSummaryScalarSummary/tagsxentropy_mean*
T0*
_output_shapes
: 
b
ScalarSummary_1/tagsConst*
dtype0*
valueB Baccuracy_mean*
_output_shapes
: 
f
ScalarSummary_1ScalarSummaryScalarSummary_1/tagsaccuracy_mean*
T0*
_output_shapes
: 
[
global-step/initial_valueConst*
dtype0*
value	B : *
_output_shapes
: 
m
global-stepVariable*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
global-step/AssignAssignglobal-stepglobal-step/initial_value*
validate_shape(*
_class
loc:@global-step*
use_locking(*
T0*
_output_shapes
: 
j
global-step/readIdentityglobal-step*
_class
loc:@global-step*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
T
gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
t
*gradients/xentropy_mean_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
$gradients/xentropy_mean_grad/ReshapeReshapegradients/Fill*gradients/xentropy_mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
e
"gradients/xentropy_mean_grad/ShapeShapeNeg*
out_type0*
T0*
_output_shapes
:
�
!gradients/xentropy_mean_grad/TileTile$gradients/xentropy_mean_grad/Reshape"gradients/xentropy_mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
g
$gradients/xentropy_mean_grad/Shape_1ShapeNeg*
out_type0*
T0*
_output_shapes
:
g
$gradients/xentropy_mean_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
l
"gradients/xentropy_mean_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
!gradients/xentropy_mean_grad/ProdProd$gradients/xentropy_mean_grad/Shape_1"gradients/xentropy_mean_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
n
$gradients/xentropy_mean_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
#gradients/xentropy_mean_grad/Prod_1Prod$gradients/xentropy_mean_grad/Shape_2$gradients/xentropy_mean_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
h
&gradients/xentropy_mean_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
$gradients/xentropy_mean_grad/MaximumMaximum#gradients/xentropy_mean_grad/Prod_1&gradients/xentropy_mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
%gradients/xentropy_mean_grad/floordivDiv!gradients/xentropy_mean_grad/Prod$gradients/xentropy_mean_grad/Maximum*
T0*
_output_shapes
: 
�
!gradients/xentropy_mean_grad/CastCast%gradients/xentropy_mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
�
$gradients/xentropy_mean_grad/truedivDiv!gradients/xentropy_mean_grad/Tile!gradients/xentropy_mean_grad/Cast*
T0*#
_output_shapes
:���������
q
gradients/Neg_grad/NegNeg$gradients/xentropy_mean_grad/truediv*
T0*#
_output_shapes
:���������
`
gradients/xentropy_grad/ShapeShapemul*
out_type0*
T0*
_output_shapes
:
^
gradients/xentropy_grad/SizeConst*
dtype0*
value	B :*
_output_shapes
: 
�
gradients/xentropy_grad/addAddxentropy/reduction_indicesgradients/xentropy_grad/Size*
T0*
_output_shapes
:
�
gradients/xentropy_grad/modModgradients/xentropy_grad/addgradients/xentropy_grad/Size*
T0*
_output_shapes
:
i
gradients/xentropy_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
e
#gradients/xentropy_grad/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
e
#gradients/xentropy_grad/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
gradients/xentropy_grad/rangeRange#gradients/xentropy_grad/range/startgradients/xentropy_grad/Size#gradients/xentropy_grad/range/delta*

Tidx0*
_output_shapes
:
d
"gradients/xentropy_grad/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: 
�
gradients/xentropy_grad/FillFillgradients/xentropy_grad/Shape_1"gradients/xentropy_grad/Fill/value*
T0*
_output_shapes
:
�
%gradients/xentropy_grad/DynamicStitchDynamicStitchgradients/xentropy_grad/rangegradients/xentropy_grad/modgradients/xentropy_grad/Shapegradients/xentropy_grad/Fill*#
_output_shapes
:���������*
T0*
N
c
!gradients/xentropy_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
gradients/xentropy_grad/MaximumMaximum%gradients/xentropy_grad/DynamicStitch!gradients/xentropy_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
 gradients/xentropy_grad/floordivDivgradients/xentropy_grad/Shapegradients/xentropy_grad/Maximum*
T0*
_output_shapes
:
�
gradients/xentropy_grad/ReshapeReshapegradients/Neg_grad/Neg%gradients/xentropy_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
gradients/xentropy_grad/TileTilegradients/xentropy_grad/Reshape gradients/xentropy_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������

c
gradients/mul_grad/ShapeShapePlaceholder*
out_type0*
T0*
_output_shapes
:
]
gradients/mul_grad/Shape_1ShapeLog*
out_type0*
T0*
_output_shapes
:
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
r
gradients/mul_grad/mulMulgradients/xentropy_grad/TileLog*
T0*'
_output_shapes
:���������

�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
|
gradients/mul_grad/mul_1MulPlaceholdergradients/xentropy_grad/Tile*
T0*'
_output_shapes
:���������

�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*'
_output_shapes
:���������
*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*-
_class#
!loc:@gradients/mul_grad/Reshape*
T0*'
_output_shapes
:���������

�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
T0*'
_output_shapes
:���������

�
gradients/Log_grad/InvInvoutput.^gradients/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������

�
gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Inv*
T0*'
_output_shapes
:���������

r
gradients/output_grad/mulMulgradients/Log_grad/muloutput*
T0*'
_output_shapes
:���������

u
+gradients/output_grad/Sum/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/output_grad/SumSumgradients/output_grad/mul+gradients/output_grad/Sum/reduction_indices*#
_output_shapes
:���������*
T0*
	keep_dims( *

Tidx0
t
#gradients/output_grad/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
gradients/output_grad/ReshapeReshapegradients/output_grad/Sum#gradients/output_grad/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients/output_grad/subSubgradients/Log_grad/mulgradients/output_grad/Reshape*
T0*'
_output_shapes
:���������

w
gradients/output_grad/mul_1Mulgradients/output_grad/suboutput*
T0*'
_output_shapes
:���������

b
gradients/add_3_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
f
gradients/add_3_grad/Shape_1Const*
dtype0*
valueB:
*
_output_shapes
:
�
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_3_grad/SumSumgradients/output_grad/mul_1*gradients/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*'
_output_shapes
:���������
*
T0*
Tshape0
�
gradients/add_3_grad/Sum_1Sumgradients/output_grad/mul_1,gradients/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
_output_shapes
:
*
T0*
Tshape0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_3_grad/Reshape*
T0*'
_output_shapes
:���������

�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*
T0*
_output_shapes
:

�
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:����������
�
 gradients/MatMul_1_grad/MatMul_1MatMuldropout_1/mul-gradients/add_3_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	�

t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0*(
_output_shapes
:����������
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0*
_output_shapes
:	�

x
"gradients/dropout_1/mul_grad/ShapeShapedropout_1/Div*
out_type0*
T0*#
_output_shapes
:���������
|
$gradients/dropout_1/mul_grad/Shape_1Shapedropout_1/Floor*
out_type0*
T0*#
_output_shapes
:���������
�
2gradients/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/mul_grad/Shape$gradients/dropout_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients/dropout_1/mul_grad/mulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout_1/Floor*
T0*
_output_shapes
:
�
 gradients/dropout_1/mul_grad/SumSum gradients/dropout_1/mul_grad/mul2gradients/dropout_1/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
$gradients/dropout_1/mul_grad/ReshapeReshape gradients/dropout_1/mul_grad/Sum"gradients/dropout_1/mul_grad/Shape*
_output_shapes
:*
T0*
Tshape0
�
"gradients/dropout_1/mul_grad/mul_1Muldropout_1/Div0gradients/MatMul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
�
"gradients/dropout_1/mul_grad/Sum_1Sum"gradients/dropout_1/mul_grad/mul_14gradients/dropout_1/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
&gradients/dropout_1/mul_grad/Reshape_1Reshape"gradients/dropout_1/mul_grad/Sum_1$gradients/dropout_1/mul_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
-gradients/dropout_1/mul_grad/tuple/group_depsNoOp%^gradients/dropout_1/mul_grad/Reshape'^gradients/dropout_1/mul_grad/Reshape_1
�
5gradients/dropout_1/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_1/mul_grad/Reshape.^gradients/dropout_1/mul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout_1/mul_grad/Reshape*
T0*
_output_shapes
:
�
7gradients/dropout_1/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_1/mul_grad/Reshape_1.^gradients/dropout_1/mul_grad/tuple/group_deps*9
_class/
-+loc:@gradients/dropout_1/mul_grad/Reshape_1*
T0*
_output_shapes
:
h
"gradients/dropout_1/Div_grad/ShapeShapeRelu_2*
out_type0*
T0*
_output_shapes
:
t
$gradients/dropout_1/Div_grad/Shape_1Shapedropout*
out_type0*
T0*#
_output_shapes
:���������
�
2gradients/dropout_1/Div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/Div_grad/Shape$gradients/dropout_1/Div_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/dropout_1/Div_grad/truedivDiv5gradients/dropout_1/mul_grad/tuple/control_dependencydropout*
T0*
_output_shapes
:
�
 gradients/dropout_1/Div_grad/SumSum$gradients/dropout_1/Div_grad/truediv2gradients/dropout_1/Div_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
$gradients/dropout_1/Div_grad/ReshapeReshape gradients/dropout_1/Div_grad/Sum"gradients/dropout_1/Div_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
b
 gradients/dropout_1/Div_grad/NegNegRelu_2*
T0*(
_output_shapes
:����������
Y
#gradients/dropout_1/Div_grad/SquareSquaredropout*
T0*
_output_shapes
:
�
&gradients/dropout_1/Div_grad/truediv_1Div gradients/dropout_1/Div_grad/Neg#gradients/dropout_1/Div_grad/Square*
T0*
_output_shapes
:
�
 gradients/dropout_1/Div_grad/mulMul5gradients/dropout_1/mul_grad/tuple/control_dependency&gradients/dropout_1/Div_grad/truediv_1*
T0*
_output_shapes
:
�
"gradients/dropout_1/Div_grad/Sum_1Sum gradients/dropout_1/Div_grad/mul4gradients/dropout_1/Div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
&gradients/dropout_1/Div_grad/Reshape_1Reshape"gradients/dropout_1/Div_grad/Sum_1$gradients/dropout_1/Div_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
-gradients/dropout_1/Div_grad/tuple/group_depsNoOp%^gradients/dropout_1/Div_grad/Reshape'^gradients/dropout_1/Div_grad/Reshape_1
�
5gradients/dropout_1/Div_grad/tuple/control_dependencyIdentity$gradients/dropout_1/Div_grad/Reshape.^gradients/dropout_1/Div_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dropout_1/Div_grad/Reshape*
T0*(
_output_shapes
:����������
�
7gradients/dropout_1/Div_grad/tuple/control_dependency_1Identity&gradients/dropout_1/Div_grad/Reshape_1.^gradients/dropout_1/Div_grad/tuple/group_deps*9
_class/
-+loc:@gradients/dropout_1/Div_grad/Reshape_1*
T0*
_output_shapes
:
�
gradients/Relu_2_grad/ReluGradReluGrad5gradients/dropout_1/Div_grad/tuple/control_dependencyRelu_2*
T0*(
_output_shapes
:����������
`
gradients/add_2_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
g
gradients/add_2_grad/Shape_1Const*
dtype0*
valueB:�*
_output_shapes
:
�
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
�
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_2_grad/Reshape*
T0*(
_output_shapes
:����������
�
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*
T0*
_output_shapes	
:�
�
gradients/MatMul_grad/MatMulMatMul-gradients/add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:����������
�
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/add_2_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
��
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0*(
_output_shapes
:����������
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
g
gradients/Reshape_1_grad/ShapeShape	MaxPool_1*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*/
_output_shapes
:���������@*
T0*
Tshape0
�
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_1	MaxPool_1 gradients/Reshape_1_grad/Reshape*/
_output_shapes
:���������@*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
gradients/Relu_1_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_1*
T0*/
_output_shapes
:���������@
b
gradients/add_1_grad/ShapeShapeConv2D_1*
out_type0*
T0*
_output_shapes
:
f
gradients/add_1_grad/Shape_1Const*
dtype0*
valueB:@*
_output_shapes
:
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*/
_output_shapes
:���������@*
T0*
Tshape0
�
gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
T0*/
_output_shapes
:���������@
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
T0*
_output_shapes
:@
d
gradients/Conv2D_1_grad/ShapeShapeMaxPool*
out_type0*
T0*
_output_shapes
:
�
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeVariable_2/read-gradients/add_1_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
x
gradients/Conv2D_1_grad/Shape_1Const*
dtype0*%
valueB"          @   *
_output_shapes
:
�
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterMaxPoolgradients/Conv2D_1_grad/Shape_1-gradients/add_1_grad/tuple/control_dependency*&
_output_shapes
: @*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
�
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:��������� 
�
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool0gradients/Conv2D_1_grad/tuple/control_dependency*/
_output_shapes
:��������� *
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
�
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:��������� 
^
gradients/add_grad/ShapeShapeConv2D*
out_type0*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
dtype0*
valueB: *
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*/
_output_shapes
:��������� *
T0*
Tshape0
�
gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*
T0*/
_output_shapes
:��������� 
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*
_output_shapes
: 
b
gradients/Conv2D_grad/ShapeShapeReshape*
out_type0*
T0*
_output_shapes
:
�
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeVariable/read+gradients/add_grad/tuple/control_dependency*J
_output_shapes8
6:4������������������������������������*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
v
gradients/Conv2D_grad/Shape_1Const*
dtype0*%
valueB"             *
_output_shapes
:
�
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/Shape_1+gradients/add_grad/tuple/control_dependency*&
_output_shapes
: *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
�
&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
�
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:���������
�
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
{
beta1_power/initial_valueConst*
dtype0*
_class
loc:@Variable*
valueB
 *fff?*
_output_shapes
: 
�
beta1_powerVariable*
	container *
_output_shapes
: *
dtype0*
shape: *
_class
loc:@Variable*
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
_class
loc:@Variable*
T0*
_output_shapes
: 
{
beta2_power/initial_valueConst*
dtype0*
_class
loc:@Variable*
valueB
 *w�?*
_output_shapes
: 
�
beta2_powerVariable*
	container *
_output_shapes
: *
dtype0*
shape: *
_class
loc:@Variable*
shared_name 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
_class
loc:@Variable*
T0*
_output_shapes
: 
j
zerosConst*
dtype0*%
valueB *    *&
_output_shapes
: 
�
Variable/AdamVariable*
	container *&
_output_shapes
: *
dtype0*
shape: *
_class
loc:@Variable*
shared_name 
�
Variable/Adam/AssignAssignVariable/Adamzeros*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
{
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*
T0*&
_output_shapes
: 
l
zeros_1Const*
dtype0*%
valueB *    *&
_output_shapes
: 
�
Variable/Adam_1Variable*
	container *&
_output_shapes
: *
dtype0*
shape: *
_class
loc:@Variable*
shared_name 
�
Variable/Adam_1/AssignAssignVariable/Adam_1zeros_1*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 

Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*
T0*&
_output_shapes
: 
T
zeros_2Const*
dtype0*
valueB *    *
_output_shapes
: 
�
Variable_1/AdamVariable*
	container *
_output_shapes
: *
dtype0*
shape: *
_class
loc:@Variable_1*
shared_name 
�
Variable_1/Adam/AssignAssignVariable_1/Adamzeros_2*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_class
loc:@Variable_1*
T0*
_output_shapes
: 
T
zeros_3Const*
dtype0*
valueB *    *
_output_shapes
: 
�
Variable_1/Adam_1Variable*
	container *
_output_shapes
: *
dtype0*
shape: *
_class
loc:@Variable_1*
shared_name 
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1zeros_3*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
T0*
_output_shapes
: 
l
zeros_4Const*
dtype0*%
valueB @*    *&
_output_shapes
: @
�
Variable_2/AdamVariable*
	container *&
_output_shapes
: @*
dtype0*
shape: @*
_class
loc:@Variable_2*
shared_name 
�
Variable_2/Adam/AssignAssignVariable_2/Adamzeros_4*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
�
Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*
T0*&
_output_shapes
: @
l
zeros_5Const*
dtype0*%
valueB @*    *&
_output_shapes
: @
�
Variable_2/Adam_1Variable*
	container *&
_output_shapes
: @*
dtype0*
shape: @*
_class
loc:@Variable_2*
shared_name 
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1zeros_5*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
�
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*
T0*&
_output_shapes
: @
T
zeros_6Const*
dtype0*
valueB@*    *
_output_shapes
:@
�
Variable_3/AdamVariable*
	container *
_output_shapes
:@*
dtype0*
shape:@*
_class
loc:@Variable_3*
shared_name 
�
Variable_3/Adam/AssignAssignVariable_3/Adamzeros_6*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
T0*
_output_shapes
:@
T
zeros_7Const*
dtype0*
valueB@*    *
_output_shapes
:@
�
Variable_3/Adam_1Variable*
	container *
_output_shapes
:@*
dtype0*
shape:@*
_class
loc:@Variable_3*
shared_name 
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1zeros_7*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
T0*
_output_shapes
:@
`
zeros_8Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Variable_4/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*
_class
loc:@Variable_4*
shared_name 
�
Variable_4/Adam/AssignAssignVariable_4/Adamzeros_8*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
{
Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��
`
zeros_9Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Variable_4/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*
_class
loc:@Variable_4*
shared_name 
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1zeros_9*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��
W
zeros_10Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Variable_5/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*
_class
loc:@Variable_5*
shared_name 
�
Variable_5/Adam/AssignAssignVariable_5/Adamzeros_10*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
v
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�
W
zeros_11Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Variable_5/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*
_class
loc:@Variable_5*
shared_name 
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1zeros_11*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�
_
zeros_12Const*
dtype0*
valueB	�
*    *
_output_shapes
:	�

�
Variable_6/AdamVariable*
	container *
_output_shapes
:	�
*
dtype0*
shape:	�
*
_class
loc:@Variable_6*
shared_name 
�
Variable_6/Adam/AssignAssignVariable_6/Adamzeros_12*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�

z
Variable_6/Adam/readIdentityVariable_6/Adam*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�

_
zeros_13Const*
dtype0*
valueB	�
*    *
_output_shapes
:	�

�
Variable_6/Adam_1Variable*
	container *
_output_shapes
:	�
*
dtype0*
shape:	�
*
_class
loc:@Variable_6*
shared_name 
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1zeros_13*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�

~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�

U
zeros_14Const*
dtype0*
valueB
*    *
_output_shapes
:

�
Variable_7/AdamVariable*
	container *
_output_shapes
:
*
dtype0*
shape:
*
_class
loc:@Variable_7*
shared_name 
�
Variable_7/Adam/AssignAssignVariable_7/Adamzeros_14*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:

u
Variable_7/Adam/readIdentityVariable_7/Adam*
_class
loc:@Variable_7*
T0*
_output_shapes
:

U
zeros_15Const*
dtype0*
valueB
*    *
_output_shapes
:

�
Variable_7/Adam_1Variable*
	container *
_output_shapes
:
*
dtype0*
shape:
*
_class
loc:@Variable_7*
shared_name 
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1zeros_15*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:

y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_class
loc:@Variable_7*
T0*
_output_shapes
:

W
Adam/learning_rateConst*
dtype0*
valueB
 *��8*
_output_shapes
: 
O

Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
_class
loc:@Variable*
use_locking( *
T0*&
_output_shapes
: 
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
_class
loc:@Variable_1*
use_locking( *
T0*
_output_shapes
: 
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_2*
use_locking( *
T0*&
_output_shapes
: @
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_3*
use_locking( *
T0*
_output_shapes
:@
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
_class
loc:@Variable_4*
use_locking( *
T0* 
_output_shapes
:
��
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
_class
loc:@Variable_5*
use_locking( *
T0*
_output_shapes	
:�
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_6*
use_locking( *
T0*
_output_shapes
:	�

�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
_class
loc:@Variable_7*
use_locking( *
T0*
_output_shapes
:

�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_class
loc:@Variable*
use_locking( *
T0*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_class
loc:@Variable*
use_locking( *
T0*
_output_shapes
: 
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
�
save/save/tensor_namesConst*
dtype0*�
value�B�BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1Bbeta1_powerBbeta2_powerBglobal-step*
_output_shapes
:
�
save/save/shapes_and_slicesConst*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:
�
	save/save
SaveSlices
save/Constsave/save/tensor_namessave/save/shapes_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1
Variable_2Variable_2/AdamVariable_2/Adam_1
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1
Variable_6Variable_6/AdamVariable_6/Adam_1
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_powerbeta2_powerglobal-step*$
T
2
{
save/control_dependencyIdentity
save/Const
^save/save*
_class
loc:@save/Const*
T0*
_output_shapes
: 
g
save/restore_slice/tensor_nameConst*
dtype0*
valueB BVariable*
_output_shapes
: 
c
"save/restore_slice/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_sliceRestoreSlice
save/Constsave/restore_slice/tensor_name"save/restore_slice/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/AssignAssignVariablesave/restore_slice*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
n
 save/restore_slice_1/tensor_nameConst*
dtype0*
valueB BVariable/Adam*
_output_shapes
: 
e
$save/restore_slice_1/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_1RestoreSlice
save/Const save/restore_slice_1/tensor_name$save/restore_slice_1/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_1AssignVariable/Adamsave/restore_slice_1*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
p
 save/restore_slice_2/tensor_nameConst*
dtype0* 
valueB BVariable/Adam_1*
_output_shapes
: 
e
$save/restore_slice_2/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_2RestoreSlice
save/Const save/restore_slice_2/tensor_name$save/restore_slice_2/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_2AssignVariable/Adam_1save/restore_slice_2*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*&
_output_shapes
: 
k
 save/restore_slice_3/tensor_nameConst*
dtype0*
valueB B
Variable_1*
_output_shapes
: 
e
$save/restore_slice_3/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_3RestoreSlice
save/Const save/restore_slice_3/tensor_name$save/restore_slice_3/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_3Assign
Variable_1save/restore_slice_3*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
p
 save/restore_slice_4/tensor_nameConst*
dtype0* 
valueB BVariable_1/Adam*
_output_shapes
: 
e
$save/restore_slice_4/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_4RestoreSlice
save/Const save/restore_slice_4/tensor_name$save/restore_slice_4/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_4AssignVariable_1/Adamsave/restore_slice_4*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
r
 save/restore_slice_5/tensor_nameConst*
dtype0*"
valueB BVariable_1/Adam_1*
_output_shapes
: 
e
$save/restore_slice_5/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_5RestoreSlice
save/Const save/restore_slice_5/tensor_name$save/restore_slice_5/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_5AssignVariable_1/Adam_1save/restore_slice_5*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes
: 
k
 save/restore_slice_6/tensor_nameConst*
dtype0*
valueB B
Variable_2*
_output_shapes
: 
e
$save/restore_slice_6/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_6RestoreSlice
save/Const save/restore_slice_6/tensor_name$save/restore_slice_6/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_6Assign
Variable_2save/restore_slice_6*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
p
 save/restore_slice_7/tensor_nameConst*
dtype0* 
valueB BVariable_2/Adam*
_output_shapes
: 
e
$save/restore_slice_7/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_7RestoreSlice
save/Const save/restore_slice_7/tensor_name$save/restore_slice_7/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_7AssignVariable_2/Adamsave/restore_slice_7*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
r
 save/restore_slice_8/tensor_nameConst*
dtype0*"
valueB BVariable_2/Adam_1*
_output_shapes
: 
e
$save/restore_slice_8/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_8RestoreSlice
save/Const save/restore_slice_8/tensor_name$save/restore_slice_8/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_8AssignVariable_2/Adam_1save/restore_slice_8*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*&
_output_shapes
: @
k
 save/restore_slice_9/tensor_nameConst*
dtype0*
valueB B
Variable_3*
_output_shapes
: 
e
$save/restore_slice_9/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_9RestoreSlice
save/Const save/restore_slice_9/tensor_name$save/restore_slice_9/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_9Assign
Variable_3save/restore_slice_9*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
q
!save/restore_slice_10/tensor_nameConst*
dtype0* 
valueB BVariable_3/Adam*
_output_shapes
: 
f
%save/restore_slice_10/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_10RestoreSlice
save/Const!save/restore_slice_10/tensor_name%save/restore_slice_10/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_10AssignVariable_3/Adamsave/restore_slice_10*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
s
!save/restore_slice_11/tensor_nameConst*
dtype0*"
valueB BVariable_3/Adam_1*
_output_shapes
: 
f
%save/restore_slice_11/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_11RestoreSlice
save/Const!save/restore_slice_11/tensor_name%save/restore_slice_11/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_11AssignVariable_3/Adam_1save/restore_slice_11*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:@
l
!save/restore_slice_12/tensor_nameConst*
dtype0*
valueB B
Variable_4*
_output_shapes
: 
f
%save/restore_slice_12/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_12RestoreSlice
save/Const!save/restore_slice_12/tensor_name%save/restore_slice_12/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_12Assign
Variable_4save/restore_slice_12*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
q
!save/restore_slice_13/tensor_nameConst*
dtype0* 
valueB BVariable_4/Adam*
_output_shapes
: 
f
%save/restore_slice_13/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_13RestoreSlice
save/Const!save/restore_slice_13/tensor_name%save/restore_slice_13/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_13AssignVariable_4/Adamsave/restore_slice_13*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
s
!save/restore_slice_14/tensor_nameConst*
dtype0*"
valueB BVariable_4/Adam_1*
_output_shapes
: 
f
%save/restore_slice_14/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_14RestoreSlice
save/Const!save/restore_slice_14/tensor_name%save/restore_slice_14/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_14AssignVariable_4/Adam_1save/restore_slice_14*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0* 
_output_shapes
:
��
l
!save/restore_slice_15/tensor_nameConst*
dtype0*
valueB B
Variable_5*
_output_shapes
: 
f
%save/restore_slice_15/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_15RestoreSlice
save/Const!save/restore_slice_15/tensor_name%save/restore_slice_15/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_15Assign
Variable_5save/restore_slice_15*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
q
!save/restore_slice_16/tensor_nameConst*
dtype0* 
valueB BVariable_5/Adam*
_output_shapes
: 
f
%save/restore_slice_16/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_16RestoreSlice
save/Const!save/restore_slice_16/tensor_name%save/restore_slice_16/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_16AssignVariable_5/Adamsave/restore_slice_16*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
s
!save/restore_slice_17/tensor_nameConst*
dtype0*"
valueB BVariable_5/Adam_1*
_output_shapes
: 
f
%save/restore_slice_17/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_17RestoreSlice
save/Const!save/restore_slice_17/tensor_name%save/restore_slice_17/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_17AssignVariable_5/Adam_1save/restore_slice_17*
validate_shape(*
_class
loc:@Variable_5*
use_locking(*
T0*
_output_shapes	
:�
l
!save/restore_slice_18/tensor_nameConst*
dtype0*
valueB B
Variable_6*
_output_shapes
: 
f
%save/restore_slice_18/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_18RestoreSlice
save/Const!save/restore_slice_18/tensor_name%save/restore_slice_18/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_18Assign
Variable_6save/restore_slice_18*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�

q
!save/restore_slice_19/tensor_nameConst*
dtype0* 
valueB BVariable_6/Adam*
_output_shapes
: 
f
%save/restore_slice_19/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_19RestoreSlice
save/Const!save/restore_slice_19/tensor_name%save/restore_slice_19/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_19AssignVariable_6/Adamsave/restore_slice_19*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�

s
!save/restore_slice_20/tensor_nameConst*
dtype0*"
valueB BVariable_6/Adam_1*
_output_shapes
: 
f
%save/restore_slice_20/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_20RestoreSlice
save/Const!save/restore_slice_20/tensor_name%save/restore_slice_20/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_20AssignVariable_6/Adam_1save/restore_slice_20*
validate_shape(*
_class
loc:@Variable_6*
use_locking(*
T0*
_output_shapes
:	�

l
!save/restore_slice_21/tensor_nameConst*
dtype0*
valueB B
Variable_7*
_output_shapes
: 
f
%save/restore_slice_21/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_21RestoreSlice
save/Const!save/restore_slice_21/tensor_name%save/restore_slice_21/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_21Assign
Variable_7save/restore_slice_21*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:

q
!save/restore_slice_22/tensor_nameConst*
dtype0* 
valueB BVariable_7/Adam*
_output_shapes
: 
f
%save/restore_slice_22/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_22RestoreSlice
save/Const!save/restore_slice_22/tensor_name%save/restore_slice_22/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_22AssignVariable_7/Adamsave/restore_slice_22*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:

s
!save/restore_slice_23/tensor_nameConst*
dtype0*"
valueB BVariable_7/Adam_1*
_output_shapes
: 
f
%save/restore_slice_23/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_23RestoreSlice
save/Const!save/restore_slice_23/tensor_name%save/restore_slice_23/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_23AssignVariable_7/Adam_1save/restore_slice_23*
validate_shape(*
_class
loc:@Variable_7*
use_locking(*
T0*
_output_shapes
:

m
!save/restore_slice_24/tensor_nameConst*
dtype0*
valueB Bbeta1_power*
_output_shapes
: 
f
%save/restore_slice_24/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_24RestoreSlice
save/Const!save/restore_slice_24/tensor_name%save/restore_slice_24/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_24Assignbeta1_powersave/restore_slice_24*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
: 
m
!save/restore_slice_25/tensor_nameConst*
dtype0*
valueB Bbeta2_power*
_output_shapes
: 
f
%save/restore_slice_25/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_25RestoreSlice
save/Const!save/restore_slice_25/tensor_name%save/restore_slice_25/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_25Assignbeta2_powersave/restore_slice_25*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
: 
m
!save/restore_slice_26/tensor_nameConst*
dtype0*
valueB Bglobal-step*
_output_shapes
: 
f
%save/restore_slice_26/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
�
save/restore_slice_26RestoreSlice
save/Const!save/restore_slice_26/tensor_name%save/restore_slice_26/shape_and_slice*
preferred_shard���������*
dt0*
_output_shapes
:
�
save/Assign_26Assignglobal-stepsave/restore_slice_26*
validate_shape(*
_class
loc:@global-step*
use_locking(*
T0*
_output_shapes
: 
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26
j
MergeSummary/MergeSummaryMergeSummaryScalarSummaryScalarSummary_1*
_output_shapes
: *
N"���9       �7�	�N����Ad*,

xentropy_mean0 ?

accuracy_means�R?s$C�:       ���	� w���A�*,

xentropy_mean�.�>

accuracy_meang?Q�b:       ���	Ԁb���A�*,

xentropy_meanY��>

accuracy_mean��l?W":E:       ���	��4���A�*,

xentropy_meanI�c>

accuracy_meanΪo?�N��:       ���	_(����A�*,

xentropy_meanj�C>

accuracy_means�r?��:       ���	|����A�*,

xentropy_meant�/>

accuracy_meantFt?qԹ:       ���	
�@���A�*,

xentropy_mean��.>

accuracy_meant?�s��:       ���	�����A�*,

xentropy_mean�m>

accuracy_mean�su?m{V:       ���	�����A�*,

xentropy_mean�U>

accuracy_mean4v?#� (:       ���	ڸ���A�*,

xentropy_meanD�>

accuracy_mean=
w?K���:       ���	Iՠ���A�*,

xentropy_meank�>

accuracy_mean�Ev?�x^u:       ���	l^V���A�	*,

xentropy_mean�w�=

accuracy_mean��w?��:       ���	����A�
*,

xentropy_mean���=

accuracy_mean�Dx?�Q�:       ���	D���A�
*,

xentropy_mean�\�=

accuracy_meanbx?�U�a:       ���	'�R��A�*,

xentropy_mean�w�=

accuracy_meanl	y?����:       ���	����A�*,

xentropy_meanr�=

accuracy_mean�Jy?=�B :       ���	�8���A�*,

xentropy_mean~��=

accuracy_mean��y?��p�:       ���	��8��A�*,

xentropy_mean[�=

accuracy_mean#�y?���:       ���	y�
 ��A�*,

xentropy_meank�=

accuracy_meanuz?vv��:       ���	���$��A�*,

xentropy_mean��=

accuracy_mean>�y?v��:       ���	���)��A�*,

xentropy_meanqʢ=

accuracy_mean#�y?_�[V:       ���	~T.��A�*,

xentropy_mean���=

accuracy_meanQz?G��:       ���	ēm3��A�*,

xentropy_meanL�=

accuracy_mean��z?��[�:       ���	F&�8��A�*,

xentropy_mean��=

accuracy_meanlxz?���U:       ���	Tq�=��A�*,

xentropy_mean�=

accuracy_mean�Cz?��d3:       ���	_�B��A�*,

xentropy_meanĊ=

accuracy_mean#J{?���v:       ���	j�CG��A�*,

xentropy_mean�e�=

accuracy_mean��{?����:       ���	��;L��A�*,

xentropy_mean�<�=

accuracy_meanvq{?C|�:       ���	E�P��A�*,

xentropy_mean�w�=

accuracy_meanvq{?�'/:       ���	��U��A�*,

xentropy_mean�ǃ=

accuracy_mean�/{?4��:       ���	�)�Z��A�*,

xentropy_meanv-�=

accuracy_mean�"{?�3q:       ���	�g_��A�*,

xentropy_mean�vu=

accuracy_meanQ�{?&�:       ���	6�5d��A�*,

xentropy_mean�j=

accuracy_mean6�{?$���:       ���	ҏ-i��A�*,

xentropy_meanT�]=

accuracy_mean-C|?��Q:       ���	��Fn��A�*,

xentropy_meanVZ=

accuracy_meanj|?ӈ�:       ���	H5�r��A�*,

xentropy_meanŲ_=

accuracy_meanQ�{?/�:       ���	P�w��A�*,

xentropy_mean)m]=

accuracy_mean�(|?��:       ���	��G|��A�*,

xentropy_mean�V=

accuracy_meanHP|?�Z�:       ���	����A�*,

xentropy_mean<
X=

accuracy_meanj|?I�y�:       ���	i�Å��A�*,

xentropy_mean�,V=

accuracy_mean-C|?��#:       ���	������A� *,

xentropy_mean�L^=

accuracy_mean6|?�;�:       ���	�Z���A� *,

xentropy_mean��_=

accuracy_mean-C|?S>z�:       ���	i5J���A�!*,

xentropy_mean+�G=

accuracy_mean�|?�!lP:       ���	#����A�"*,

xentropy_mean�\=

accuracy_mean�|?w���:       ���	�����A�#*,

xentropy_meanvS=

accuracy_mean�(|?�;v�:       ���	v�y���A�#*,

xentropy_meanN=

accuracy_mean�|?�z:       ���	�>���A�$*,

xentropy_mean\�C=

accuracy_mean?�|?]�2�:       ���	�[W���A�%*,

xentropy_mean��5=

accuracy_mean��|?�3 �:       ���	�����A�&*,

xentropy_mean*�7=

accuracy_mean�}?�3:       ���	e�����A�'*,

xentropy_mean@:=

accuracy_mean$�|?i��E:       ���	Q�>���A�'*,

xentropy_mean1�6=

accuracy_mean[�|?���:       ���	�u���A�(*,

xentropy_meanO1=

accuracy_mean?�|?<y�c:       ���	"7����A�)*,

xentropy_meanÓ5=

accuracy_mean��|?,f�:       ���	x�\���A�**,

xentropy_mean_'=

accuracy_mean�!}?�:       ���	RF����A�**,

xentropy_mean/7!=

accuracy_meanmV}?N��:       ���	|+����A�+*,

xentropy_mean�27=

accuracy_mean��|?�݋:       ���	b�����A�,*,

xentropy_mean־0=

accuracy_mean$�|?�fyz:       ���	?�����A�-*,

xentropy_meanA%-=

accuracy_mean[�|?�/o�:       ���	����A�.*,

xentropy_meanO�+=

accuracy_mean�!}?����:       ���	�$����A�.*,

xentropy_mean�g&=

accuracy_mean�c}?��}�:       ���	B]D���A�/*,

xentropy_mean�5F=

accuracy_mean-C|?��]:       ���	�����A�0*,

xentropy_mean��@=

accuracy_mean��|?Geƿ:       ���	�1����A�1*,

xentropy_mean�N%=

accuracy_mean�}?��$�:       ���	|�|���A�2*,

xentropy_mean'=

accuracy_meanۊ}?��!p:       ���	�1����A�2*,

xentropy_mean"=

accuracy_mean6<}?���:       ���	��2 ��A�3*,

xentropy_mean}�=

accuracy_meanRI}?� �:       ���	�C���A�4*,

xentropy_mean�D=

accuracy_mean�}}?�8	J:       ���	q"	��A�5*,

xentropy_mean4'=

accuracy_meanv�|?�0�:       ���	����A�5*,

xentropy_mean�2=

accuracy_mean�p}?O�B:       ���	�E���A�6*,

xentropy_meanoV=

accuracy_mean��}?��[�:       ���	2>���A�7*,

xentropy_meanE =

accuracy_mean6<}?h:       ���	�FL��A�8*,

xentropy_mean��=

accuracy_mean��}?�ٴ:       ���	m� ��A�9*,

xentropy_mean*=

accuracy_mean�c}?AȽ*:       ���	�A�$��A�9*,

xentropy_mean�%!=

accuracy_mean6<}?:^�:       ���	��)��A�:*,

xentropy_mean��=

accuracy_meanRI}?��9}:       ���	�;�-��A�;*,

xentropy_mean�o=

accuracy_meanRI}?�#�:       ���	��\2��A�<*,

xentropy_meanQ�=

accuracy_mean�}}?o���:       ���	���6��A�<*,

xentropy_mean/�=

accuracy_mean�}?H��{:       ���	�Gr;��A�=*,

xentropy_mean0=

accuracy_mean��}?d���:       ���	||�?��A�>*,

xentropy_meanL�=

accuracy_mean��}?q+@:       ���	PwD��A�?*,

xentropy_meanp�=

accuracy_meanmV}?���<:       ���	@MI��A�@*,

xentropy_mean�
=

accuracy_mean�}?E�9-:       ���	Nd�M��A�@*,

xentropy_mean��=

accuracy_mean/}?zP�q:       ���	��MR��A�A*,

xentropy_mean�=

accuracy_mean�}}?�|c:       ���	Oy W��A�B*,

xentropy_mean��=

accuracy_mean��}?���:       ���	r3r[��A�C*,

xentropy_meandt=

accuracy_mean��}?���:       ���	.`��A�C*,

xentropy_mean�O=

accuracy_mean�}?����:       ���	�Яd��A�D*,

xentropy_mean��=

accuracy_mean�c}?�@�:       ���	�Q�i��A�E*,

xentropy_mean�=

accuracy_mean�c}?5��:       ���	��ln��A�F*,

xentropy_mean�r=

accuracy_mean�}?5�q:       ���	$`�r��A�G*,

xentropy_mean��=

accuracy_mean�c}?<�-:       ���	�Ȣw��A�G*,

xentropy_mean�=

accuracy_mean-�}?���	:       ���	.�t|��A�H*,

xentropy_mean	�=

accuracy_mean-�}?tI�:       ���	����A�I*,

xentropy_mean��=

accuracy_mean��}?PǖO:       ���	\j���A�J*,

xentropy_meanA=

accuracy_meanH�}?'U�:       ���	���A�K*,

xentropy_mean��=

accuracy_mean/}?..�:       ���	$*.���A�K*,

xentropy_meanڿ=

accuracy_mean-�}?>�3�:       ���	*r���A�L*,

xentropy_mean h=

accuracy_mean?5~?q�j:       ���	A���A�M*,

xentropy_mean�~=

accuracy_mean�}}?*���:       ���	��D���A�N*,

xentropy_meany
=

accuracy_meanH�}?��:       ���	e�;���A�N*,

xentropy_meanqX=

accuracy_mean��}?�o�:       ���	��ש��A�O*,

xentropy_meanV�=

accuracy_mean	~?��k:       ���	�zK���A�P*,

xentropy_mean�T
=

accuracy_meanۊ}?Nb�:       ���	�Ŵ��A�Q*,

xentropy_mean�� =

accuracy_mean�}?��21:       ���	�v���A�R*,

xentropy_mean��=

accuracy_mean-�}?D��c:       ���	�B���A�R*,

xentropy_mean6��<

accuracy_mean�}?��:       ���	�P���A�S*,

xentropy_mean��=

accuracy_mean�}??�~:       ���	�q���A�T*,

xentropy_mean�=

accuracy_mean��}?
_z�:       ���	�����A�U*,

xentropy_mean6��<

accuracy_mean� ~?_ϵV:       ���	�A����A�U*,

xentropy_mean�� =

accuracy_mean��}?8 ��:       ���	�����A�V*,

xentropy_mean�-	=

accuracy_mean��}?��m:       ���	sٍ���A�W*,

xentropy_mean�=�<

accuracy_mean�~?�gw:       ���	ߣ����A�X*,

xentropy_mean�M�<

accuracy_mean�}?ב@\:       ���	^����A�Y*,

xentropy_mean���<

accuracy_mean� ~?�Ѳ<:       ���	wg,���A�Y*,

xentropy_meanw�<

accuracy_mean�~?����:       ���	�ڝ���A�Z*,

xentropy_mean�L�<

accuracy_mean$(~?��$,:       ���	�3}���A�[*,

xentropy_meanKX�<

accuracy_mean?5~?Q6O:       ���	s�o ��A�\*,

xentropy_meanZ=

accuracy_mean��}?M��:       ���	�cd��A�\*,

xentropy_mean�w
=

accuracy_mean-�}?eoҔ:       ���	}�J
��A�]*,

xentropy_mean��	=

accuracy_meand�}?x^��:       ���	��t��A�^*,

xentropy_mean���<

accuracy_mean	~?��ۃ:       ���	�����A�_*,

xentropy_mean�f=

accuracy_mean��}?ΦW:       ���	����A�`*,

xentropy_meanAI=

accuracy_mean-�}?ɋ�u:       ���	s���A�`*,

xentropy_mean|��<

accuracy_meand�}?�8{�:       ���	���"��A�a*,

xentropy_mean1=

accuracy_mean�}?Aq��:       ���	��X(��A�b*,

xentropy_mean��=

accuracy_mean��}?����:       ���	E�n-��A�c*,

xentropy_meanFA=

accuracy_mean-�}?�1%�:       ���	�x92��A�d*,

xentropy_mean  =

accuracy_meanH�}?۸�:       ���	�07��A�d*,

xentropy_mean��=

accuracy_meanۊ}?��#:       ���	��<��A�e*,

xentropy_mean/�=

accuracy_mean�}?�|?Z:       ���	��@��A�f*,

xentropy_mean��<

accuracy_meanH�}?��::       ���	8�F��A�g*,

xentropy_meanc�<

accuracy_mean�}?*0#:       ���	ĲdK��A�g*,

xentropy_mean$=

accuracy_mean�}?N#�m:       ���	�I�P��A�h*,

xentropy_mean�I=

accuracy_mean�}?=]�:       ���	�V��A�i*,

xentropy_mean��=

accuracy_meanH�}?���:       ���	!YP[��A�j*,

xentropy_mean�{=

accuracy_mean-�}?I���:       ���	��8`��A�k*,

xentropy_mean.8=

accuracy_meanH�}?�"А:       ���	4�Ce��A�k*,

xentropy_mean�x�<

accuracy_mean� ~?���:       ���	wi
j��A�l*,

xentropy_meanr=

accuracy_mean��}?E�Vd:       ���	��o��A�m*,

xentropy_meanF��<

accuracy_mean�~?��)g:       ���	Jt��A�n*,

xentropy_meanY�=

accuracy_mean�c}?�8�<:       ���	��y��A�n*,

xentropy_meanC=

accuracy_mean��}?eN��:       ���	���}��A�o*,

xentropy_mean�?�<

accuracy_mean��}?�O��:       ���	��ς��A�p*,

xentropy_mean9E�<

accuracy_mean�}?��|:       ���	�χ��A�q*,

xentropy_mean)��<

accuracy_mean��}?A��F:       ���	~5���A�r*,

xentropy_mean� =

accuracy_mean�}?"Ae:       ���	��;���A�r*,

xentropy_mean�A�<

accuracy_mean� ~?��R�:       ���	e�D���A�s*,

xentropy_mean�� =

accuracy_mean�~?J`:       ���	𢿜��A�t*,

xentropy_mean�L�<

accuracy_mean��}?=�m�:       ���	4�i���A�u*,

xentropy_mean��=

accuracy_meand�}?8���:       ���	)#����A�u*,

xentropy_meanV�	=

accuracy_mean-�}?�x�:       ���	GL���A�v*,

xentropy_mean�i=

accuracy_meand�}?���':       ���	d���A�w*,

xentropy_mean��	=

accuracy_mean�}?졆/:       ���	zÖ���A�x*,

xentropy_meanԖ�<

accuracy_mean�}?8���:       ���	��^���A�y*,

xentropy_meanC��<

accuracy_mean�~?w��S:       ���	ZJ���A�y*,

xentropy_meana�<

accuracy_mean	~?�̫u:       ���	h�����A�z*,

xentropy_mean��=

accuracy_mean�~?ͺ�:       ���	}�K���A�{*,

xentropy_meanH#=

accuracy_mean-�}?�w��:       ���	�.����A�|*,

xentropy_meanT�=

accuracy_meanmV}?!�x�:       ���	2w����A�}*,

xentropy_mean�O=

accuracy_meand�}?�K4�:       ���	��?���A�}*,

xentropy_mean�=

accuracy_meanH�}?�:1:       ���	_!���A�~*,

xentropy_meanQ&�<

accuracy_mean��}?��׸:       ���	�����A�*,

xentropy_mean��=

accuracy_mean�}?��0�;       #�\	1�*���A��*,

xentropy_mean���<

accuracy_mean�}?�=Ҽ;       #�\	�����A�*,

xentropy_mean��<

accuracy_mean��}?4,>�;       #�\	�B���A؁*,

xentropy_meant�<

accuracy_mean�}?�u�;       #�\	������A��*,

xentropy_meanW =

accuracy_mean��}?d���;       #�\	�����A��*,

xentropy_meanj�=

accuracy_meanۊ}?���;       #�\	������A��*,

xentropy_mean�T =

accuracy_mean�}?��;       #�\	����A�*,

xentropy_mean�� =

accuracy_mean	~?�R��;       #�\	l���A̅*,

xentropy_mean��=

accuracy_mean��}?�C;       #�\	����A��*,

xentropy_mean=�=

accuracy_mean��}?�@��;       #�\	U���A��*,

xentropy_mean|�=

accuracy_mean�}?�8)(;       #�\	��h��A��*,

xentropy_mean=�=

accuracy_mean��}?j���;       #�\	y�y��A܈*,

xentropy_mean8�=

accuracy_mean�}?+�D;       #�\	�U<"��A��*,

xentropy_mean�8�<

accuracy_mean$(~?A�I?;       #�\	]1[(��A��*,

xentropy_mean�5�<

accuracy_mean?5~?�	2;       #�\	.��A��*,

xentropy_mean��<

accuracy_mean[B~?�N9Y;       #�\	I��3��A�*,

xentropy_mean��<

accuracy_meanvO~?����;       #�\	�e(9��AЌ*,

xentropy_meanM;=

accuracy_mean-�}?dɉs;       #�\	q��>��A��*,

xentropy_mean��<

accuracy_meanvO~?\��};       #�\	��iD��A��*,

xentropy_mean-�<

accuracy_mean�i~?"�<[;       #�\	��N��A��*,

xentropy_mean�R�<

accuracy_mean$(~?���;       #�\	V��V��A��*,

xentropy_mean��<

accuracy_mean[B~?����;       #�\	���^��AĐ*,

xentropy_mean��=

accuracy_mean��}?v�t;       #�\	���f��A��*,

xentropy_meant��<

accuracy_mean�}?(Ƶ1;       #�\	�-o��A��*,

xentropy_mean(;=

accuracy_meanH�}?��6(;       #�\	�%Kw��A�*,

xentropy_mean��<

accuracy_meanH�}?�L�;       #�\	=���Aԓ*,

xentropy_mean��=

accuracy_mean�~?��9�;       #�\	�r����A��*,

xentropy_mean+ =

accuracy_mean�~?U�7;       #�\	��Č��A��*,

xentropy_meano*=

accuracy_mean��}?�;       #�\	�t����A��*,

xentropy_meanR�	=

accuracy_mean�~?�#�O;       #�\	Ί����A�*,

xentropy_mean���<

accuracy_mean�v~?
�z�;       #�\	�꣪��Aȗ*,

xentropy_mean��<

accuracy_mean	~?���;       #�\	3k����A��*,

xentropy_meanR7�<

accuracy_meanvO~?���;       #�\	&8����A��*,

xentropy_meanV��<

accuracy_mean[B~?h2;       #�\	�����A��*,

xentropy_mean���<

accuracy_mean�~?�
.�;       #�\	��H���Aؚ*,

xentropy_mean��=

accuracy_mean�}?���N;       #�\	�����A��*,

xentropy_mean2�<

accuracy_mean[B~?�y�;       #�\	t)���A��*,

xentropy_mean��=

accuracy_mean	~?�� 