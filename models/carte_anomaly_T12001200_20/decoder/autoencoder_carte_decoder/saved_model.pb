��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02unknown8��
�
conv2d_transpose_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_18/bias
�
,conv2d_transpose_18/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_18/bias*
_output_shapes
:*
dtype0
�
conv2d_transpose_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_18/kernel
�
.conv2d_transpose_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_18/kernel*&
_output_shapes
: *
dtype0
�
conv2d_transpose_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_17/bias
�
,conv2d_transpose_17/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_17/bias*
_output_shapes
: *
dtype0
�
conv2d_transpose_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_17/kernel
�
.conv2d_transpose_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_17/kernel*&
_output_shapes
: @*
dtype0
�
conv2d_transpose_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_16/bias
�
,conv2d_transpose_16/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_16/bias*
_output_shapes
:@*
dtype0
�
conv2d_transpose_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*+
shared_nameconv2d_transpose_16/kernel
�
.conv2d_transpose_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_16/kernel*'
_output_shapes
:@�*
dtype0
�
conv2d_transpose_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameconv2d_transpose_15/bias
�
,conv2d_transpose_15/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_15/bias*
_output_shapes	
:�*
dtype0
�
conv2d_transpose_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameconv2d_transpose_15/kernel
�
.conv2d_transpose_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_15/kernel*(
_output_shapes
:��*
dtype0
�
serving_default_input_10Placeholder*0
_output_shapes
:����������*
dtype0*%
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10conv2d_transpose_15/kernelconv2d_transpose_15/biasconv2d_transpose_16/kernelconv2d_transpose_16/biasconv2d_transpose_17/kernelconv2d_transpose_17/biasconv2d_transpose_18/kernelconv2d_transpose_18/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_117777

NoOpNoOp
�!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�!
value�!B�! B�!
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op*
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op*
<
0
1
2
3
&4
'5
/6
07*
<
0
1
2
3
&4
'5
/6
07*
* 
�
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
7trace_0
8trace_1
9trace_2
:trace_3* 
6
;trace_0
<trace_1
=trace_2
>trace_3* 
* 

?serving_default* 

0
1*

0
1*
* 
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Etrace_0* 

Ftrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_15/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_15/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ltrace_0* 

Mtrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_16/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_16/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

&0
'1*

&0
'1*
* 
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

Strace_0* 

Ttrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_17/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_17/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

/0
01*

/0
01*
* 
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

Ztrace_0* 

[trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_18/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_18/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
'
0
1
2
3
4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.conv2d_transpose_15/kernel/Read/ReadVariableOp,conv2d_transpose_15/bias/Read/ReadVariableOp.conv2d_transpose_16/kernel/Read/ReadVariableOp,conv2d_transpose_16/bias/Read/ReadVariableOp.conv2d_transpose_17/kernel/Read/ReadVariableOp,conv2d_transpose_17/bias/Read/ReadVariableOp.conv2d_transpose_18/kernel/Read/ReadVariableOp,conv2d_transpose_18/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_118206
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_transpose_15/kernelconv2d_transpose_15/biasconv2d_transpose_16/kernelconv2d_transpose_16/biasconv2d_transpose_17/kernelconv2d_transpose_17/biasconv2d_transpose_18/kernelconv2d_transpose_18/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_118240��
�

�
(__inference_decoder_layer_call_fn_117706
input_10#
unknown:��
	unknown_0:	�$
	unknown_1:@�
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_117666y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:����������
"
_user_specified_name
input_10
�!
�
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_117475

inputsC
(conv2d_transpose_readvariableop_resource:@�-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�!
�
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_118159

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
4__inference_conv2d_transpose_16_layer_call_fn_118039

inputs"
unknown:@�
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_117475�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
4__inference_conv2d_transpose_17_layer_call_fn_118082

inputs!
unknown: @
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_117520�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

�
(__inference_decoder_layer_call_fn_117619
input_10#
unknown:��
	unknown_0:	�$
	unknown_1:@�
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_117600y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:����������
"
_user_specified_name
input_10
�!
�
O__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_117430

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������k
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
__inference__traced_save_118206
file_prefix9
5savev2_conv2d_transpose_15_kernel_read_readvariableop7
3savev2_conv2d_transpose_15_bias_read_readvariableop9
5savev2_conv2d_transpose_16_kernel_read_readvariableop7
3savev2_conv2d_transpose_16_bias_read_readvariableop9
5savev2_conv2d_transpose_17_kernel_read_readvariableop7
3savev2_conv2d_transpose_17_bias_read_readvariableop9
5savev2_conv2d_transpose_18_kernel_read_readvariableop7
3savev2_conv2d_transpose_18_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_conv2d_transpose_15_kernel_read_readvariableop3savev2_conv2d_transpose_15_bias_read_readvariableop5savev2_conv2d_transpose_16_kernel_read_readvariableop3savev2_conv2d_transpose_16_bias_read_readvariableop5savev2_conv2d_transpose_17_kernel_read_readvariableop3savev2_conv2d_transpose_17_bias_read_readvariableop5savev2_conv2d_transpose_18_kernel_read_readvariableop3savev2_conv2d_transpose_18_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*{
_input_shapesj
h: :��:�:@�:@: @: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:.*
(
_output_shapes
:��:!

_output_shapes	
:�:-)
'
_output_shapes
:@�: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::	

_output_shapes
: 
�
�
4__inference_conv2d_transpose_15_layer_call_fn_117996

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_117430�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�!
�
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_117565

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
C__inference_decoder_layer_call_and_return_conditional_losses_117600

inputs6
conv2d_transpose_15_117579:��)
conv2d_transpose_15_117581:	�5
conv2d_transpose_16_117584:@�(
conv2d_transpose_16_117586:@4
conv2d_transpose_17_117589: @(
conv2d_transpose_17_117591: 4
conv2d_transpose_18_117594: (
conv2d_transpose_18_117596:
identity��+conv2d_transpose_15/StatefulPartitionedCall�+conv2d_transpose_16/StatefulPartitionedCall�+conv2d_transpose_17/StatefulPartitionedCall�+conv2d_transpose_18/StatefulPartitionedCall�
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_15_117579conv2d_transpose_15_117581*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_117430�
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0conv2d_transpose_16_117584conv2d_transpose_16_117586*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_117475�
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0conv2d_transpose_17_117589conv2d_transpose_17_117591*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������hh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_117520�
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0conv2d_transpose_18_117594conv2d_transpose_18_117596*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_117565�
IdentityIdentity4conv2d_transpose_18/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp,^conv2d_transpose_15/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : 2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
O__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_118030

inputsD
(conv2d_transpose_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :�y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������k
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
4__inference_conv2d_transpose_18_layer_call_fn_118125

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_117565�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�!
�
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_118116

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
C__inference_decoder_layer_call_and_return_conditional_losses_117754
input_106
conv2d_transpose_15_117733:��)
conv2d_transpose_15_117735:	�5
conv2d_transpose_16_117738:@�(
conv2d_transpose_16_117740:@4
conv2d_transpose_17_117743: @(
conv2d_transpose_17_117745: 4
conv2d_transpose_18_117748: (
conv2d_transpose_18_117750:
identity��+conv2d_transpose_15/StatefulPartitionedCall�+conv2d_transpose_16/StatefulPartitionedCall�+conv2d_transpose_17/StatefulPartitionedCall�+conv2d_transpose_18/StatefulPartitionedCall�
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCallinput_10conv2d_transpose_15_117733conv2d_transpose_15_117735*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_117430�
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0conv2d_transpose_16_117738conv2d_transpose_16_117740*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_117475�
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0conv2d_transpose_17_117743conv2d_transpose_17_117745*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������hh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_117520�
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0conv2d_transpose_18_117748conv2d_transpose_18_117750*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_117565�
IdentityIdentity4conv2d_transpose_18/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp,^conv2d_transpose_15/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : 2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall:Z V
0
_output_shapes
:����������
"
_user_specified_name
input_10
�}
�	
!__inference__wrapped_model_117392
input_10`
Ddecoder_conv2d_transpose_15_conv2d_transpose_readvariableop_resource:��J
;decoder_conv2d_transpose_15_biasadd_readvariableop_resource:	�_
Ddecoder_conv2d_transpose_16_conv2d_transpose_readvariableop_resource:@�I
;decoder_conv2d_transpose_16_biasadd_readvariableop_resource:@^
Ddecoder_conv2d_transpose_17_conv2d_transpose_readvariableop_resource: @I
;decoder_conv2d_transpose_17_biasadd_readvariableop_resource: ^
Ddecoder_conv2d_transpose_18_conv2d_transpose_readvariableop_resource: I
;decoder_conv2d_transpose_18_biasadd_readvariableop_resource:
identity��2decoder/conv2d_transpose_15/BiasAdd/ReadVariableOp�;decoder/conv2d_transpose_15/conv2d_transpose/ReadVariableOp�2decoder/conv2d_transpose_16/BiasAdd/ReadVariableOp�;decoder/conv2d_transpose_16/conv2d_transpose/ReadVariableOp�2decoder/conv2d_transpose_17/BiasAdd/ReadVariableOp�;decoder/conv2d_transpose_17/conv2d_transpose/ReadVariableOp�2decoder/conv2d_transpose_18/BiasAdd/ReadVariableOp�;decoder/conv2d_transpose_18/conv2d_transpose/ReadVariableOpY
!decoder/conv2d_transpose_15/ShapeShapeinput_10*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)decoder/conv2d_transpose_15/strided_sliceStridedSlice*decoder/conv2d_transpose_15/Shape:output:08decoder/conv2d_transpose_15/strided_slice/stack:output:0:decoder/conv2d_transpose_15/strided_slice/stack_1:output:0:decoder/conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :f
#decoder/conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
!decoder/conv2d_transpose_15/stackPack2decoder/conv2d_transpose_15/strided_slice:output:0,decoder/conv2d_transpose_15/stack/1:output:0,decoder/conv2d_transpose_15/stack/2:output:0,decoder/conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+decoder/conv2d_transpose_15/strided_slice_1StridedSlice*decoder/conv2d_transpose_15/stack:output:0:decoder/conv2d_transpose_15/strided_slice_1/stack:output:0<decoder/conv2d_transpose_15/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
;decoder/conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_15_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
,decoder/conv2d_transpose_15/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_15/stack:output:0Cdecoder/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0input_10*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
2decoder/conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#decoder/conv2d_transpose_15/BiasAddBiasAdd5decoder/conv2d_transpose_15/conv2d_transpose:output:0:decoder/conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
 decoder/conv2d_transpose_15/ReluRelu,decoder/conv2d_transpose_15/BiasAdd:output:0*
T0*0
_output_shapes
:����������
!decoder/conv2d_transpose_16/ShapeShape.decoder/conv2d_transpose_15/Relu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)decoder/conv2d_transpose_16/strided_sliceStridedSlice*decoder/conv2d_transpose_16/Shape:output:08decoder/conv2d_transpose_16/strided_slice/stack:output:0:decoder/conv2d_transpose_16/strided_slice/stack_1:output:0:decoder/conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :4e
#decoder/conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :4e
#decoder/conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
!decoder/conv2d_transpose_16/stackPack2decoder/conv2d_transpose_16/strided_slice:output:0,decoder/conv2d_transpose_16/stack/1:output:0,decoder/conv2d_transpose_16/stack/2:output:0,decoder/conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+decoder/conv2d_transpose_16/strided_slice_1StridedSlice*decoder/conv2d_transpose_16/stack:output:0:decoder/conv2d_transpose_16/strided_slice_1/stack:output:0<decoder/conv2d_transpose_16/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
;decoder/conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_16_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
,decoder/conv2d_transpose_16/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_16/stack:output:0Cdecoder/conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_15/Relu:activations:0*
T0*/
_output_shapes
:���������44@*
paddingSAME*
strides
�
2decoder/conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#decoder/conv2d_transpose_16/BiasAddBiasAdd5decoder/conv2d_transpose_16/conv2d_transpose:output:0:decoder/conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������44@�
 decoder/conv2d_transpose_16/ReluRelu,decoder/conv2d_transpose_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������44@
!decoder/conv2d_transpose_17/ShapeShape.decoder/conv2d_transpose_16/Relu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)decoder/conv2d_transpose_17/strided_sliceStridedSlice*decoder/conv2d_transpose_17/Shape:output:08decoder/conv2d_transpose_17/strided_slice/stack:output:0:decoder/conv2d_transpose_17/strided_slice/stack_1:output:0:decoder/conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :he
#decoder/conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :he
#decoder/conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
!decoder/conv2d_transpose_17/stackPack2decoder/conv2d_transpose_17/strided_slice:output:0,decoder/conv2d_transpose_17/stack/1:output:0,decoder/conv2d_transpose_17/stack/2:output:0,decoder/conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+decoder/conv2d_transpose_17/strided_slice_1StridedSlice*decoder/conv2d_transpose_17/stack:output:0:decoder/conv2d_transpose_17/strided_slice_1/stack:output:0<decoder/conv2d_transpose_17/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
;decoder/conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_17_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
,decoder/conv2d_transpose_17/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_17/stack:output:0Cdecoder/conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_16/Relu:activations:0*
T0*/
_output_shapes
:���������hh *
paddingSAME*
strides
�
2decoder/conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
#decoder/conv2d_transpose_17/BiasAddBiasAdd5decoder/conv2d_transpose_17/conv2d_transpose:output:0:decoder/conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������hh �
 decoder/conv2d_transpose_17/ReluRelu,decoder/conv2d_transpose_17/BiasAdd:output:0*
T0*/
_output_shapes
:���������hh 
!decoder/conv2d_transpose_18/ShapeShape.decoder/conv2d_transpose_17/Relu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)decoder/conv2d_transpose_18/strided_sliceStridedSlice*decoder/conv2d_transpose_18/Shape:output:08decoder/conv2d_transpose_18/strided_slice/stack:output:0:decoder/conv2d_transpose_18/strided_slice/stack_1:output:0:decoder/conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#decoder/conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�f
#decoder/conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�e
#decoder/conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
!decoder/conv2d_transpose_18/stackPack2decoder/conv2d_transpose_18/strided_slice:output:0,decoder/conv2d_transpose_18/stack/1:output:0,decoder/conv2d_transpose_18/stack/2:output:0,decoder/conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+decoder/conv2d_transpose_18/strided_slice_1StridedSlice*decoder/conv2d_transpose_18/stack:output:0:decoder/conv2d_transpose_18/strided_slice_1/stack:output:0<decoder/conv2d_transpose_18/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
;decoder/conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
,decoder/conv2d_transpose_18/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_18/stack:output:0Cdecoder/conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_17/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
2decoder/conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#decoder/conv2d_transpose_18/BiasAddBiasAdd5decoder/conv2d_transpose_18/conv2d_transpose:output:0:decoder/conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
#decoder/conv2d_transpose_18/SigmoidSigmoid,decoder/conv2d_transpose_18/BiasAdd:output:0*
T0*1
_output_shapes
:������������
IdentityIdentity'decoder/conv2d_transpose_18/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp3^decoder/conv2d_transpose_15/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_15/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_16/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_16/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_17/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_17/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_18/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_18/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : 2h
2decoder/conv2d_transpose_15/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_15/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_15/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_15/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_16/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_16/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_16/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_16/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_17/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_17/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_17/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_17/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_18/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_18/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_18/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_18/conv2d_transpose/ReadVariableOp:Z V
0
_output_shapes
:����������
"
_user_specified_name
input_10
�q
�
C__inference_decoder_layer_call_and_return_conditional_losses_117903

inputsX
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource:��B
3conv2d_transpose_15_biasadd_readvariableop_resource:	�W
<conv2d_transpose_16_conv2d_transpose_readvariableop_resource:@�A
3conv2d_transpose_16_biasadd_readvariableop_resource:@V
<conv2d_transpose_17_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_17_biasadd_readvariableop_resource: V
<conv2d_transpose_18_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_18_biasadd_readvariableop_resource:
identity��*conv2d_transpose_15/BiasAdd/ReadVariableOp�3conv2d_transpose_15/conv2d_transpose/ReadVariableOp�*conv2d_transpose_16/BiasAdd/ReadVariableOp�3conv2d_transpose_16/conv2d_transpose/ReadVariableOp�*conv2d_transpose_17/BiasAdd/ReadVariableOp�3conv2d_transpose_17/conv2d_transpose/ReadVariableOp�*conv2d_transpose_18/BiasAdd/ReadVariableOp�3conv2d_transpose_18/conv2d_transpose/ReadVariableOpO
conv2d_transpose_15/ShapeShapeinputs*
T0*
_output_shapes
:q
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
*conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_15/BiasAddBiasAdd-conv2d_transpose_15/conv2d_transpose:output:02conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
conv2d_transpose_15/ReluRelu$conv2d_transpose_15/BiasAdd:output:0*
T0*0
_output_shapes
:����������o
conv2d_transpose_16/ShapeShape&conv2d_transpose_15/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!conv2d_transpose_16/strided_sliceStridedSlice"conv2d_transpose_16/Shape:output:00conv2d_transpose_16/strided_slice/stack:output:02conv2d_transpose_16/strided_slice/stack_1:output:02conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :4]
conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :4]
conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
conv2d_transpose_16/stackPack*conv2d_transpose_16/strided_slice:output:0$conv2d_transpose_16/stack/1:output:0$conv2d_transpose_16/stack/2:output:0$conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#conv2d_transpose_16/strided_slice_1StridedSlice"conv2d_transpose_16/stack:output:02conv2d_transpose_16/strided_slice_1/stack:output:04conv2d_transpose_16/strided_slice_1/stack_1:output:04conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
3conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_16_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
$conv2d_transpose_16/conv2d_transposeConv2DBackpropInput"conv2d_transpose_16/stack:output:0;conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_15/Relu:activations:0*
T0*/
_output_shapes
:���������44@*
paddingSAME*
strides
�
*conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_transpose_16/BiasAddBiasAdd-conv2d_transpose_16/conv2d_transpose:output:02conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������44@�
conv2d_transpose_16/ReluRelu$conv2d_transpose_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������44@o
conv2d_transpose_17/ShapeShape&conv2d_transpose_16/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!conv2d_transpose_17/strided_sliceStridedSlice"conv2d_transpose_17/Shape:output:00conv2d_transpose_17/strided_slice/stack:output:02conv2d_transpose_17/strided_slice/stack_1:output:02conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :h]
conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :h]
conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_17/stackPack*conv2d_transpose_17/strided_slice:output:0$conv2d_transpose_17/stack/1:output:0$conv2d_transpose_17/stack/2:output:0$conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#conv2d_transpose_17/strided_slice_1StridedSlice"conv2d_transpose_17/stack:output:02conv2d_transpose_17/strided_slice_1/stack:output:04conv2d_transpose_17/strided_slice_1/stack_1:output:04conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
3conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_17_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
$conv2d_transpose_17/conv2d_transposeConv2DBackpropInput"conv2d_transpose_17/stack:output:0;conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_16/Relu:activations:0*
T0*/
_output_shapes
:���������hh *
paddingSAME*
strides
�
*conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_transpose_17/BiasAddBiasAdd-conv2d_transpose_17/conv2d_transpose:output:02conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������hh �
conv2d_transpose_17/ReluRelu$conv2d_transpose_17/BiasAdd:output:0*
T0*/
_output_shapes
:���������hh o
conv2d_transpose_18/ShapeShape&conv2d_transpose_17/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!conv2d_transpose_18/strided_sliceStridedSlice"conv2d_transpose_18/Shape:output:00conv2d_transpose_18/strided_slice/stack:output:02conv2d_transpose_18/strided_slice/stack_1:output:02conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�^
conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_18/stackPack*conv2d_transpose_18/strided_slice:output:0$conv2d_transpose_18/stack/1:output:0$conv2d_transpose_18/stack/2:output:0$conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#conv2d_transpose_18/strided_slice_1StridedSlice"conv2d_transpose_18/stack:output:02conv2d_transpose_18/strided_slice_1/stack:output:04conv2d_transpose_18/strided_slice_1/stack_1:output:04conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
3conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
$conv2d_transpose_18/conv2d_transposeConv2DBackpropInput"conv2d_transpose_18/stack:output:0;conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_17/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
*conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_18/BiasAddBiasAdd-conv2d_transpose_18/conv2d_transpose:output:02conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
conv2d_transpose_18/SigmoidSigmoid$conv2d_transpose_18/BiasAdd:output:0*
T0*1
_output_shapes
:�����������x
IdentityIdentityconv2d_transpose_18/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp+^conv2d_transpose_15/BiasAdd/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp+^conv2d_transpose_16/BiasAdd/ReadVariableOp4^conv2d_transpose_16/conv2d_transpose/ReadVariableOp+^conv2d_transpose_17/BiasAdd/ReadVariableOp4^conv2d_transpose_17/conv2d_transpose/ReadVariableOp+^conv2d_transpose_18/BiasAdd/ReadVariableOp4^conv2d_transpose_18/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : 2X
*conv2d_transpose_15/BiasAdd/ReadVariableOp*conv2d_transpose_15/BiasAdd/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_16/BiasAdd/ReadVariableOp*conv2d_transpose_16/BiasAdd/ReadVariableOp2j
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp3conv2d_transpose_16/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_17/BiasAdd/ReadVariableOp*conv2d_transpose_17/BiasAdd/ReadVariableOp2j
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp3conv2d_transpose_17/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_18/BiasAdd/ReadVariableOp*conv2d_transpose_18/BiasAdd/ReadVariableOp2j
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp3conv2d_transpose_18/conv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
$__inference_signature_wrapper_117777
input_10#
unknown:��
	unknown_0:	�$
	unknown_1:@�
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_117392y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:����������
"
_user_specified_name
input_10
�%
�
"__inference__traced_restore_118240
file_prefixG
+assignvariableop_conv2d_transpose_15_kernel:��:
+assignvariableop_1_conv2d_transpose_15_bias:	�H
-assignvariableop_2_conv2d_transpose_16_kernel:@�9
+assignvariableop_3_conv2d_transpose_16_bias:@G
-assignvariableop_4_conv2d_transpose_17_kernel: @9
+assignvariableop_5_conv2d_transpose_17_bias: G
-assignvariableop_6_conv2d_transpose_18_kernel: 9
+assignvariableop_7_conv2d_transpose_18_bias:

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp+assignvariableop_conv2d_transpose_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_conv2d_transpose_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp-assignvariableop_2_conv2d_transpose_16_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp+assignvariableop_3_conv2d_transpose_16_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp-assignvariableop_4_conv2d_transpose_17_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_conv2d_transpose_17_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp-assignvariableop_6_conv2d_transpose_18_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_conv2d_transpose_18_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_9IdentityIdentity_8:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*"
_acd_function_control_output(*
_output_shapes
 "!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
C__inference_decoder_layer_call_and_return_conditional_losses_117730
input_106
conv2d_transpose_15_117709:��)
conv2d_transpose_15_117711:	�5
conv2d_transpose_16_117714:@�(
conv2d_transpose_16_117716:@4
conv2d_transpose_17_117719: @(
conv2d_transpose_17_117721: 4
conv2d_transpose_18_117724: (
conv2d_transpose_18_117726:
identity��+conv2d_transpose_15/StatefulPartitionedCall�+conv2d_transpose_16/StatefulPartitionedCall�+conv2d_transpose_17/StatefulPartitionedCall�+conv2d_transpose_18/StatefulPartitionedCall�
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCallinput_10conv2d_transpose_15_117709conv2d_transpose_15_117711*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_117430�
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0conv2d_transpose_16_117714conv2d_transpose_16_117716*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_117475�
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0conv2d_transpose_17_117719conv2d_transpose_17_117721*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������hh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_117520�
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0conv2d_transpose_18_117724conv2d_transpose_18_117726*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_117565�
IdentityIdentity4conv2d_transpose_18/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp,^conv2d_transpose_15/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : 2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall:Z V
0
_output_shapes
:����������
"
_user_specified_name
input_10
�!
�
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_117520

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�q
�
C__inference_decoder_layer_call_and_return_conditional_losses_117987

inputsX
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource:��B
3conv2d_transpose_15_biasadd_readvariableop_resource:	�W
<conv2d_transpose_16_conv2d_transpose_readvariableop_resource:@�A
3conv2d_transpose_16_biasadd_readvariableop_resource:@V
<conv2d_transpose_17_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_17_biasadd_readvariableop_resource: V
<conv2d_transpose_18_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_18_biasadd_readvariableop_resource:
identity��*conv2d_transpose_15/BiasAdd/ReadVariableOp�3conv2d_transpose_15/conv2d_transpose/ReadVariableOp�*conv2d_transpose_16/BiasAdd/ReadVariableOp�3conv2d_transpose_16/conv2d_transpose/ReadVariableOp�*conv2d_transpose_17/BiasAdd/ReadVariableOp�3conv2d_transpose_17/conv2d_transpose/ReadVariableOp�*conv2d_transpose_18/BiasAdd/ReadVariableOp�3conv2d_transpose_18/conv2d_transpose/ReadVariableOpO
conv2d_transpose_15/ShapeShapeinputs*
T0*
_output_shapes
:q
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value
B :��
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
*conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_transpose_15/BiasAddBiasAdd-conv2d_transpose_15/conv2d_transpose:output:02conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
conv2d_transpose_15/ReluRelu$conv2d_transpose_15/BiasAdd:output:0*
T0*0
_output_shapes
:����������o
conv2d_transpose_16/ShapeShape&conv2d_transpose_15/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!conv2d_transpose_16/strided_sliceStridedSlice"conv2d_transpose_16/Shape:output:00conv2d_transpose_16/strided_slice/stack:output:02conv2d_transpose_16/strided_slice/stack_1:output:02conv2d_transpose_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_16/stack/1Const*
_output_shapes
: *
dtype0*
value	B :4]
conv2d_transpose_16/stack/2Const*
_output_shapes
: *
dtype0*
value	B :4]
conv2d_transpose_16/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@�
conv2d_transpose_16/stackPack*conv2d_transpose_16/strided_slice:output:0$conv2d_transpose_16/stack/1:output:0$conv2d_transpose_16/stack/2:output:0$conv2d_transpose_16/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#conv2d_transpose_16/strided_slice_1StridedSlice"conv2d_transpose_16/stack:output:02conv2d_transpose_16/strided_slice_1/stack:output:04conv2d_transpose_16/strided_slice_1/stack_1:output:04conv2d_transpose_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
3conv2d_transpose_16/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_16_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
$conv2d_transpose_16/conv2d_transposeConv2DBackpropInput"conv2d_transpose_16/stack:output:0;conv2d_transpose_16/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_15/Relu:activations:0*
T0*/
_output_shapes
:���������44@*
paddingSAME*
strides
�
*conv2d_transpose_16/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_transpose_16/BiasAddBiasAdd-conv2d_transpose_16/conv2d_transpose:output:02conv2d_transpose_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������44@�
conv2d_transpose_16/ReluRelu$conv2d_transpose_16/BiasAdd:output:0*
T0*/
_output_shapes
:���������44@o
conv2d_transpose_17/ShapeShape&conv2d_transpose_16/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!conv2d_transpose_17/strided_sliceStridedSlice"conv2d_transpose_17/Shape:output:00conv2d_transpose_17/strided_slice/stack:output:02conv2d_transpose_17/strided_slice/stack_1:output:02conv2d_transpose_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_17/stack/1Const*
_output_shapes
: *
dtype0*
value	B :h]
conv2d_transpose_17/stack/2Const*
_output_shapes
: *
dtype0*
value	B :h]
conv2d_transpose_17/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_17/stackPack*conv2d_transpose_17/strided_slice:output:0$conv2d_transpose_17/stack/1:output:0$conv2d_transpose_17/stack/2:output:0$conv2d_transpose_17/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#conv2d_transpose_17/strided_slice_1StridedSlice"conv2d_transpose_17/stack:output:02conv2d_transpose_17/strided_slice_1/stack:output:04conv2d_transpose_17/strided_slice_1/stack_1:output:04conv2d_transpose_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
3conv2d_transpose_17/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_17_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
$conv2d_transpose_17/conv2d_transposeConv2DBackpropInput"conv2d_transpose_17/stack:output:0;conv2d_transpose_17/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_16/Relu:activations:0*
T0*/
_output_shapes
:���������hh *
paddingSAME*
strides
�
*conv2d_transpose_17/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_transpose_17/BiasAddBiasAdd-conv2d_transpose_17/conv2d_transpose:output:02conv2d_transpose_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������hh �
conv2d_transpose_17/ReluRelu$conv2d_transpose_17/BiasAdd:output:0*
T0*/
_output_shapes
:���������hh o
conv2d_transpose_18/ShapeShape&conv2d_transpose_17/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
!conv2d_transpose_18/strided_sliceStridedSlice"conv2d_transpose_18/Shape:output:00conv2d_transpose_18/strided_slice/stack:output:02conv2d_transpose_18/strided_slice/stack_1:output:02conv2d_transpose_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_18/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�^
conv2d_transpose_18/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�]
conv2d_transpose_18/stack/3Const*
_output_shapes
: *
dtype0*
value	B :�
conv2d_transpose_18/stackPack*conv2d_transpose_18/strided_slice:output:0$conv2d_transpose_18/stack/1:output:0$conv2d_transpose_18/stack/2:output:0$conv2d_transpose_18/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#conv2d_transpose_18/strided_slice_1StridedSlice"conv2d_transpose_18/stack:output:02conv2d_transpose_18/strided_slice_1/stack:output:04conv2d_transpose_18/strided_slice_1/stack_1:output:04conv2d_transpose_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
3conv2d_transpose_18/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_18_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0�
$conv2d_transpose_18/conv2d_transposeConv2DBackpropInput"conv2d_transpose_18/stack:output:0;conv2d_transpose_18/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_17/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
*conv2d_transpose_18/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_transpose_18/BiasAddBiasAdd-conv2d_transpose_18/conv2d_transpose:output:02conv2d_transpose_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
conv2d_transpose_18/SigmoidSigmoid$conv2d_transpose_18/BiasAdd:output:0*
T0*1
_output_shapes
:�����������x
IdentityIdentityconv2d_transpose_18/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp+^conv2d_transpose_15/BiasAdd/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp+^conv2d_transpose_16/BiasAdd/ReadVariableOp4^conv2d_transpose_16/conv2d_transpose/ReadVariableOp+^conv2d_transpose_17/BiasAdd/ReadVariableOp4^conv2d_transpose_17/conv2d_transpose/ReadVariableOp+^conv2d_transpose_18/BiasAdd/ReadVariableOp4^conv2d_transpose_18/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : 2X
*conv2d_transpose_15/BiasAdd/ReadVariableOp*conv2d_transpose_15/BiasAdd/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_16/BiasAdd/ReadVariableOp*conv2d_transpose_16/BiasAdd/ReadVariableOp2j
3conv2d_transpose_16/conv2d_transpose/ReadVariableOp3conv2d_transpose_16/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_17/BiasAdd/ReadVariableOp*conv2d_transpose_17/BiasAdd/ReadVariableOp2j
3conv2d_transpose_17/conv2d_transpose/ReadVariableOp3conv2d_transpose_17/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_18/BiasAdd/ReadVariableOp*conv2d_transpose_18/BiasAdd/ReadVariableOp2j
3conv2d_transpose_18/conv2d_transpose/ReadVariableOp3conv2d_transpose_18/conv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
(__inference_decoder_layer_call_fn_117798

inputs#
unknown:��
	unknown_0:	�$
	unknown_1:@�
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_117600y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_118073

inputsC
(conv2d_transpose_readvariableop_resource:@�-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
C__inference_decoder_layer_call_and_return_conditional_losses_117666

inputs6
conv2d_transpose_15_117645:��)
conv2d_transpose_15_117647:	�5
conv2d_transpose_16_117650:@�(
conv2d_transpose_16_117652:@4
conv2d_transpose_17_117655: @(
conv2d_transpose_17_117657: 4
conv2d_transpose_18_117660: (
conv2d_transpose_18_117662:
identity��+conv2d_transpose_15/StatefulPartitionedCall�+conv2d_transpose_16/StatefulPartitionedCall�+conv2d_transpose_17/StatefulPartitionedCall�+conv2d_transpose_18/StatefulPartitionedCall�
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_15_117645conv2d_transpose_15_117647*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_117430�
+conv2d_transpose_16/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_15/StatefulPartitionedCall:output:0conv2d_transpose_16_117650conv2d_transpose_16_117652*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_117475�
+conv2d_transpose_17/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_16/StatefulPartitionedCall:output:0conv2d_transpose_17_117655conv2d_transpose_17_117657*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������hh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_117520�
+conv2d_transpose_18/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_17/StatefulPartitionedCall:output:0conv2d_transpose_18_117660conv2d_transpose_18_117662*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_117565�
IdentityIdentity4conv2d_transpose_18/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp,^conv2d_transpose_15/StatefulPartitionedCall,^conv2d_transpose_16/StatefulPartitionedCall,^conv2d_transpose_17/StatefulPartitionedCall,^conv2d_transpose_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : 2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall2Z
+conv2d_transpose_16/StatefulPartitionedCall+conv2d_transpose_16/StatefulPartitionedCall2Z
+conv2d_transpose_17/StatefulPartitionedCall+conv2d_transpose_17/StatefulPartitionedCall2Z
+conv2d_transpose_18/StatefulPartitionedCall+conv2d_transpose_18/StatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
(__inference_decoder_layer_call_fn_117819

inputs#
unknown:��
	unknown_0:	�$
	unknown_1:@�
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_117666y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
F
input_10:
serving_default_input_10:0����������Q
conv2d_transpose_18:
StatefulPartitionedCall:0�����������tensorflow/serving/predict:�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op"
_tf_keras_layer
X
0
1
2
3
&4
'5
/6
07"
trackable_list_wrapper
X
0
1
2
3
&4
'5
/6
07"
trackable_list_wrapper
 "
trackable_list_wrapper
�
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
7trace_0
8trace_1
9trace_2
:trace_32�
(__inference_decoder_layer_call_fn_117619
(__inference_decoder_layer_call_fn_117798
(__inference_decoder_layer_call_fn_117819
(__inference_decoder_layer_call_fn_117706�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z7trace_0z8trace_1z9trace_2z:trace_3
�
;trace_0
<trace_1
=trace_2
>trace_32�
C__inference_decoder_layer_call_and_return_conditional_losses_117903
C__inference_decoder_layer_call_and_return_conditional_losses_117987
C__inference_decoder_layer_call_and_return_conditional_losses_117730
C__inference_decoder_layer_call_and_return_conditional_losses_117754�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z;trace_0z<trace_1z=trace_2z>trace_3
�B�
!__inference__wrapped_model_117392input_10"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
?serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Etrace_02�
4__inference_conv2d_transpose_15_layer_call_fn_117996�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zEtrace_0
�
Ftrace_02�
O__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_118030�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zFtrace_0
6:4��2conv2d_transpose_15/kernel
':%�2conv2d_transpose_15/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ltrace_02�
4__inference_conv2d_transpose_16_layer_call_fn_118039�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0
�
Mtrace_02�
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_118073�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zMtrace_0
5:3@�2conv2d_transpose_16/kernel
&:$@2conv2d_transpose_16/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
Strace_02�
4__inference_conv2d_transpose_17_layer_call_fn_118082�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zStrace_0
�
Ttrace_02�
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_118116�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zTtrace_0
4:2 @2conv2d_transpose_17/kernel
&:$ 2conv2d_transpose_17/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
Ztrace_02�
4__inference_conv2d_transpose_18_layer_call_fn_118125�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0
�
[trace_02�
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_118159�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z[trace_0
4:2 2conv2d_transpose_18/kernel
&:$2conv2d_transpose_18/bias
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_decoder_layer_call_fn_117619input_10"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_decoder_layer_call_fn_117798inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_decoder_layer_call_fn_117819inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_decoder_layer_call_fn_117706input_10"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_decoder_layer_call_and_return_conditional_losses_117903inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_decoder_layer_call_and_return_conditional_losses_117987inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_decoder_layer_call_and_return_conditional_losses_117730input_10"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_decoder_layer_call_and_return_conditional_losses_117754input_10"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_117777input_10"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_conv2d_transpose_15_layer_call_fn_117996inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_118030inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_conv2d_transpose_16_layer_call_fn_118039inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_118073inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_conv2d_transpose_17_layer_call_fn_118082inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_118116inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_conv2d_transpose_18_layer_call_fn_118125inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_118159inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_117392�&'/0:�7
0�-
+�(
input_10����������
� "S�P
N
conv2d_transpose_187�4
conv2d_transpose_18������������
O__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_118030�J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
4__inference_conv2d_transpose_15_layer_call_fn_117996�J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
O__inference_conv2d_transpose_16_layer_call_and_return_conditional_losses_118073�J�G
@�=
;�8
inputs,����������������������������
� "?�<
5�2
0+���������������������������@
� �
4__inference_conv2d_transpose_16_layer_call_fn_118039�J�G
@�=
;�8
inputs,����������������������������
� "2�/+���������������������������@�
O__inference_conv2d_transpose_17_layer_call_and_return_conditional_losses_118116�&'I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+��������������������������� 
� �
4__inference_conv2d_transpose_17_layer_call_fn_118082�&'I�F
?�<
:�7
inputs+���������������������������@
� "2�/+��������������������������� �
O__inference_conv2d_transpose_18_layer_call_and_return_conditional_losses_118159�/0I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������
� �
4__inference_conv2d_transpose_18_layer_call_fn_118125�/0I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+����������������������������
C__inference_decoder_layer_call_and_return_conditional_losses_117730&'/0B�?
8�5
+�(
input_10����������
p 

 
� "/�,
%�"
0�����������
� �
C__inference_decoder_layer_call_and_return_conditional_losses_117754&'/0B�?
8�5
+�(
input_10����������
p

 
� "/�,
%�"
0�����������
� �
C__inference_decoder_layer_call_and_return_conditional_losses_117903}&'/0@�=
6�3
)�&
inputs����������
p 

 
� "/�,
%�"
0�����������
� �
C__inference_decoder_layer_call_and_return_conditional_losses_117987}&'/0@�=
6�3
)�&
inputs����������
p

 
� "/�,
%�"
0�����������
� �
(__inference_decoder_layer_call_fn_117619r&'/0B�?
8�5
+�(
input_10����������
p 

 
� ""�������������
(__inference_decoder_layer_call_fn_117706r&'/0B�?
8�5
+�(
input_10����������
p

 
� ""�������������
(__inference_decoder_layer_call_fn_117798p&'/0@�=
6�3
)�&
inputs����������
p 

 
� ""�������������
(__inference_decoder_layer_call_fn_117819p&'/0@�=
6�3
)�&
inputs����������
p

 
� ""�������������
$__inference_signature_wrapper_117777�&'/0F�C
� 
<�9
7
input_10+�(
input_10����������"S�P
N
conv2d_transpose_187�4
conv2d_transpose_18�����������