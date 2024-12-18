��
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
Conv2D

input"T
filter"T
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
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
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
H
ShardedFilename
basename	
shard

num_shards
filename
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
 �"serve*2.10.02unknown8��

u
conv2d_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_69/bias
n
"conv2d_69/bias/Read/ReadVariableOpReadVariableOpconv2d_69/bias*
_output_shapes	
:�*
dtype0
�
conv2d_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_69/kernel

$conv2d_69/kernel/Read/ReadVariableOpReadVariableOpconv2d_69/kernel*(
_output_shapes
:��*
dtype0
�
&batch_normalization_91/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_91/moving_variance
�
:batch_normalization_91/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_91/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_91/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_91/moving_mean
�
6batch_normalization_91/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_91/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_91/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_91/beta
�
/batch_normalization_91/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_91/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_91/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_91/gamma
�
0batch_normalization_91/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_91/gamma*
_output_shapes	
:�*
dtype0
u
conv2d_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_68/bias
n
"conv2d_68/bias/Read/ReadVariableOpReadVariableOpconv2d_68/bias*
_output_shapes	
:�*
dtype0
�
conv2d_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*!
shared_nameconv2d_68/kernel
~
$conv2d_68/kernel/Read/ReadVariableOpReadVariableOpconv2d_68/kernel*'
_output_shapes
:@�*
dtype0
�
&batch_normalization_90/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_90/moving_variance
�
:batch_normalization_90/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_90/moving_variance*
_output_shapes
:@*
dtype0
�
"batch_normalization_90/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_90/moving_mean
�
6batch_normalization_90/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_90/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_90/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_90/beta
�
/batch_normalization_90/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_90/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_90/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_90/gamma
�
0batch_normalization_90/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_90/gamma*
_output_shapes
:@*
dtype0
t
conv2d_67/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_67/bias
m
"conv2d_67/bias/Read/ReadVariableOpReadVariableOpconv2d_67/bias*
_output_shapes
:@*
dtype0
�
conv2d_67/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_67/kernel
}
$conv2d_67/kernel/Read/ReadVariableOpReadVariableOpconv2d_67/kernel*&
_output_shapes
: @*
dtype0
�
&batch_normalization_89/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_89/moving_variance
�
:batch_normalization_89/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_89/moving_variance*
_output_shapes
: *
dtype0
�
"batch_normalization_89/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_89/moving_mean
�
6batch_normalization_89/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_89/moving_mean*
_output_shapes
: *
dtype0
�
batch_normalization_89/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_89/beta
�
/batch_normalization_89/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_89/beta*
_output_shapes
: *
dtype0
�
batch_normalization_89/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_89/gamma
�
0batch_normalization_89/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_89/gamma*
_output_shapes
: *
dtype0
t
conv2d_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_66/bias
m
"conv2d_66/bias/Read/ReadVariableOpReadVariableOpconv2d_66/bias*
_output_shapes
: *
dtype0
�
conv2d_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_66/kernel
}
$conv2d_66/kernel/Read/ReadVariableOpReadVariableOpconv2d_66/kernel*&
_output_shapes
: *
dtype0
�
serving_default_input_16Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_16conv2d_66/kernelconv2d_66/biasbatch_normalization_89/gammabatch_normalization_89/beta"batch_normalization_89/moving_mean&batch_normalization_89/moving_varianceconv2d_67/kernelconv2d_67/biasbatch_normalization_90/gammabatch_normalization_90/beta"batch_normalization_90/moving_mean&batch_normalization_90/moving_varianceconv2d_68/kernelconv2d_68/biasbatch_normalization_91/gammabatch_normalization_91/beta"batch_normalization_91/moving_mean&batch_normalization_91/moving_varianceconv2d_69/kernelconv2d_69/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_173832

NoOpNoOp
�J
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�J
value�JB�J B�J
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#axis
	$gamma
%beta
&moving_mean
'moving_variance*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias
 6_jit_compiled_convolution_op*
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=axis
	>gamma
?beta
@moving_mean
Amoving_variance*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses* 
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias
 P_jit_compiled_convolution_op*
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance*
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses* 
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias
 j_jit_compiled_convolution_op*
�
0
1
$2
%3
&4
'5
46
57
>8
?9
@10
A11
N12
O13
X14
Y15
Z16
[17
h18
i19*
j
0
1
$2
%3
44
55
>6
?7
N8
O9
X10
Y11
h12
i13*
* 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ptrace_0
qtrace_1
rtrace_2
strace_3* 
6
ttrace_0
utrace_1
vtrace_2
wtrace_3* 
* 

xserving_default* 

0
1*

0
1*
* 
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

~trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv2d_66/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_66/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
$0
%1
&2
'3*

$0
%1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_89/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_89/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_89/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_89/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

40
51*

40
51*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_67/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_67/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
>0
?1
@2
A3*

>0
?1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_90/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_90/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_90/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_90/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

N0
O1*

N0
O1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_68/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_68/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
X0
Y1
Z2
[3*

X0
Y1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_91/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_91/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_91/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_91/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

h0
i1*

h0
i1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_69/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_69/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
&0
'1
@2
A3
Z4
[5*
R
0
1
2
3
4
5
6
7
	8

9
10*
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

&0
'1*
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

@0
A1*
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

Z0
[1*
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
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_66/kernel/Read/ReadVariableOp"conv2d_66/bias/Read/ReadVariableOp0batch_normalization_89/gamma/Read/ReadVariableOp/batch_normalization_89/beta/Read/ReadVariableOp6batch_normalization_89/moving_mean/Read/ReadVariableOp:batch_normalization_89/moving_variance/Read/ReadVariableOp$conv2d_67/kernel/Read/ReadVariableOp"conv2d_67/bias/Read/ReadVariableOp0batch_normalization_90/gamma/Read/ReadVariableOp/batch_normalization_90/beta/Read/ReadVariableOp6batch_normalization_90/moving_mean/Read/ReadVariableOp:batch_normalization_90/moving_variance/Read/ReadVariableOp$conv2d_68/kernel/Read/ReadVariableOp"conv2d_68/bias/Read/ReadVariableOp0batch_normalization_91/gamma/Read/ReadVariableOp/batch_normalization_91/beta/Read/ReadVariableOp6batch_normalization_91/moving_mean/Read/ReadVariableOp:batch_normalization_91/moving_variance/Read/ReadVariableOp$conv2d_69/kernel/Read/ReadVariableOp"conv2d_69/bias/Read/ReadVariableOpConst*!
Tin
2*
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
__inference__traced_save_174455
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_66/kernelconv2d_66/biasbatch_normalization_89/gammabatch_normalization_89/beta"batch_normalization_89/moving_mean&batch_normalization_89/moving_varianceconv2d_67/kernelconv2d_67/biasbatch_normalization_90/gammabatch_normalization_90/beta"batch_normalization_90/moving_mean&batch_normalization_90/moving_varianceconv2d_68/kernelconv2d_68/biasbatch_normalization_91/gammabatch_normalization_91/beta"batch_normalization_91/moving_mean&batch_normalization_91/moving_varianceconv2d_69/kernelconv2d_69/bias* 
Tin
2*
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
"__inference__traced_restore_174525��	
�6
�	
C__inference_encoder_layer_call_and_return_conditional_losses_173785
input_16*
conv2d_66_173734: 
conv2d_66_173736: +
batch_normalization_89_173739: +
batch_normalization_89_173741: +
batch_normalization_89_173743: +
batch_normalization_89_173745: *
conv2d_67_173749: @
conv2d_67_173751:@+
batch_normalization_90_173754:@+
batch_normalization_90_173756:@+
batch_normalization_90_173758:@+
batch_normalization_90_173760:@+
conv2d_68_173764:@�
conv2d_68_173766:	�,
batch_normalization_91_173769:	�,
batch_normalization_91_173771:	�,
batch_normalization_91_173773:	�,
batch_normalization_91_173775:	�,
conv2d_69_173779:��
conv2d_69_173781:	�
identity��.batch_normalization_89/StatefulPartitionedCall�.batch_normalization_90/StatefulPartitionedCall�.batch_normalization_91/StatefulPartitionedCall�!conv2d_66/StatefulPartitionedCall�!conv2d_67/StatefulPartitionedCall�!conv2d_68/StatefulPartitionedCall�!conv2d_69/StatefulPartitionedCall�
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCallinput_16conv2d_66_173734conv2d_66_173736*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_66_layer_call_and_return_conditional_losses_173281�
.batch_normalization_89/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_89_173739batch_normalization_89_173741batch_normalization_89_173743batch_normalization_89_173745*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_173124�
leaky_re_lu_85/PartitionedCallPartitionedCall7batch_normalization_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_85_layer_call_and_return_conditional_losses_173301�
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_85/PartitionedCall:output:0conv2d_67_173749conv2d_67_173751*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_67_layer_call_and_return_conditional_losses_173314�
.batch_normalization_90/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_90_173754batch_normalization_90_173756batch_normalization_90_173758batch_normalization_90_173760*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_173188�
leaky_re_lu_86/PartitionedCallPartitionedCall7batch_normalization_90/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_86_layer_call_and_return_conditional_losses_173334�
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_86/PartitionedCall:output:0conv2d_68_173764conv2d_68_173766*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_68_layer_call_and_return_conditional_losses_173347�
.batch_normalization_91/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_91_173769batch_normalization_91_173771batch_normalization_91_173773batch_normalization_91_173775*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_173252�
leaky_re_lu_87/PartitionedCallPartitionedCall7batch_normalization_91/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_87_layer_call_and_return_conditional_losses_173367�
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_87/PartitionedCall:output:0conv2d_69_173779conv2d_69_173781*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_69_layer_call_and_return_conditional_losses_173380�
IdentityIdentity*conv2d_69/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp/^batch_normalization_89/StatefulPartitionedCall/^batch_normalization_90/StatefulPartitionedCall/^batch_normalization_91/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:�����������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_89/StatefulPartitionedCall.batch_normalization_89/StatefulPartitionedCall2`
.batch_normalization_90/StatefulPartitionedCall.batch_normalization_90/StatefulPartitionedCall2`
.batch_normalization_91/StatefulPartitionedCall.batch_normalization_91/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
input_16
�6
�	
C__inference_encoder_layer_call_and_return_conditional_losses_173387

inputs*
conv2d_66_173282: 
conv2d_66_173284: +
batch_normalization_89_173287: +
batch_normalization_89_173289: +
batch_normalization_89_173291: +
batch_normalization_89_173293: *
conv2d_67_173315: @
conv2d_67_173317:@+
batch_normalization_90_173320:@+
batch_normalization_90_173322:@+
batch_normalization_90_173324:@+
batch_normalization_90_173326:@+
conv2d_68_173348:@�
conv2d_68_173350:	�,
batch_normalization_91_173353:	�,
batch_normalization_91_173355:	�,
batch_normalization_91_173357:	�,
batch_normalization_91_173359:	�,
conv2d_69_173381:��
conv2d_69_173383:	�
identity��.batch_normalization_89/StatefulPartitionedCall�.batch_normalization_90/StatefulPartitionedCall�.batch_normalization_91/StatefulPartitionedCall�!conv2d_66/StatefulPartitionedCall�!conv2d_67/StatefulPartitionedCall�!conv2d_68/StatefulPartitionedCall�!conv2d_69/StatefulPartitionedCall�
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_66_173282conv2d_66_173284*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_66_layer_call_and_return_conditional_losses_173281�
.batch_normalization_89/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_89_173287batch_normalization_89_173289batch_normalization_89_173291batch_normalization_89_173293*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_173093�
leaky_re_lu_85/PartitionedCallPartitionedCall7batch_normalization_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_85_layer_call_and_return_conditional_losses_173301�
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_85/PartitionedCall:output:0conv2d_67_173315conv2d_67_173317*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_67_layer_call_and_return_conditional_losses_173314�
.batch_normalization_90/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_90_173320batch_normalization_90_173322batch_normalization_90_173324batch_normalization_90_173326*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_173157�
leaky_re_lu_86/PartitionedCallPartitionedCall7batch_normalization_90/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_86_layer_call_and_return_conditional_losses_173334�
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_86/PartitionedCall:output:0conv2d_68_173348conv2d_68_173350*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_68_layer_call_and_return_conditional_losses_173347�
.batch_normalization_91/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_91_173353batch_normalization_91_173355batch_normalization_91_173357batch_normalization_91_173359*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_173221�
leaky_re_lu_87/PartitionedCallPartitionedCall7batch_normalization_91/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_87_layer_call_and_return_conditional_losses_173367�
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_87/PartitionedCall:output:0conv2d_69_173381conv2d_69_173383*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_69_layer_call_and_return_conditional_losses_173380�
IdentityIdentity*conv2d_69/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp/^batch_normalization_89/StatefulPartitionedCall/^batch_normalization_90/StatefulPartitionedCall/^batch_normalization_91/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:�����������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_89/StatefulPartitionedCall.batch_normalization_89/StatefulPartitionedCall2`
.batch_normalization_90/StatefulPartitionedCall.batch_normalization_90/StatefulPartitionedCall2`
.batch_normalization_91/StatefulPartitionedCall.batch_normalization_91/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_68_layer_call_fn_174269

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_68_layer_call_and_return_conditional_losses_173347x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������  �`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
�
(__inference_encoder_layer_call_fn_173922

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�&

unknown_17:��

unknown_18:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_173589x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:�����������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
f
J__inference_leaky_re_lu_86_layer_call_and_return_conditional_losses_173334

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������@@@*
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@@:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
f
J__inference_leaky_re_lu_85_layer_call_and_return_conditional_losses_173301

inputs
identitya
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:����������� *
alpha%���>i
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:����������� :Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_173221

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�3
�	
__inference__traced_save_174455
file_prefix/
+savev2_conv2d_66_kernel_read_readvariableop-
)savev2_conv2d_66_bias_read_readvariableop;
7savev2_batch_normalization_89_gamma_read_readvariableop:
6savev2_batch_normalization_89_beta_read_readvariableopA
=savev2_batch_normalization_89_moving_mean_read_readvariableopE
Asavev2_batch_normalization_89_moving_variance_read_readvariableop/
+savev2_conv2d_67_kernel_read_readvariableop-
)savev2_conv2d_67_bias_read_readvariableop;
7savev2_batch_normalization_90_gamma_read_readvariableop:
6savev2_batch_normalization_90_beta_read_readvariableopA
=savev2_batch_normalization_90_moving_mean_read_readvariableopE
Asavev2_batch_normalization_90_moving_variance_read_readvariableop/
+savev2_conv2d_68_kernel_read_readvariableop-
)savev2_conv2d_68_bias_read_readvariableop;
7savev2_batch_normalization_91_gamma_read_readvariableop:
6savev2_batch_normalization_91_beta_read_readvariableopA
=savev2_batch_normalization_91_moving_mean_read_readvariableopE
Asavev2_batch_normalization_91_moving_variance_read_readvariableop/
+savev2_conv2d_69_kernel_read_readvariableop-
)savev2_conv2d_69_bias_read_readvariableop
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
: �

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_66_kernel_read_readvariableop)savev2_conv2d_66_bias_read_readvariableop7savev2_batch_normalization_89_gamma_read_readvariableop6savev2_batch_normalization_89_beta_read_readvariableop=savev2_batch_normalization_89_moving_mean_read_readvariableopAsavev2_batch_normalization_89_moving_variance_read_readvariableop+savev2_conv2d_67_kernel_read_readvariableop)savev2_conv2d_67_bias_read_readvariableop7savev2_batch_normalization_90_gamma_read_readvariableop6savev2_batch_normalization_90_beta_read_readvariableop=savev2_batch_normalization_90_moving_mean_read_readvariableopAsavev2_batch_normalization_90_moving_variance_read_readvariableop+savev2_conv2d_68_kernel_read_readvariableop)savev2_conv2d_68_bias_read_readvariableop7savev2_batch_normalization_91_gamma_read_readvariableop6savev2_batch_normalization_91_beta_read_readvariableop=savev2_batch_normalization_91_moving_mean_read_readvariableopAsavev2_batch_normalization_91_moving_variance_read_readvariableop+savev2_conv2d_69_kernel_read_readvariableop)savev2_conv2d_69_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
2�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : : : @:@:@:@:@:@:@�:�:�:�:�:�:��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:

_output_shapes
: 
�
K
/__inference_leaky_re_lu_85_layer_call_fn_174163

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_85_layer_call_and_return_conditional_losses_173301j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:����������� :Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
�
E__inference_conv2d_69_layer_call_and_return_conditional_losses_173380

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
�
E__inference_conv2d_66_layer_call_and_return_conditional_losses_173281

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:����������� k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:����������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_69_layer_call_and_return_conditional_losses_174372

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_174140

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_91_layer_call_fn_174293

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_173221�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
f
J__inference_leaky_re_lu_85_layer_call_and_return_conditional_losses_174168

inputs
identitya
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:����������� *
alpha%���>i
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:����������� :Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_173124

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_173252

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_173188

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
(__inference_encoder_layer_call_fn_173430
input_16!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�&

unknown_17:��

unknown_18:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_173387x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:�����������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
input_16
�T
�
"__inference__traced_restore_174525
file_prefix;
!assignvariableop_conv2d_66_kernel: /
!assignvariableop_1_conv2d_66_bias: =
/assignvariableop_2_batch_normalization_89_gamma: <
.assignvariableop_3_batch_normalization_89_beta: C
5assignvariableop_4_batch_normalization_89_moving_mean: G
9assignvariableop_5_batch_normalization_89_moving_variance: =
#assignvariableop_6_conv2d_67_kernel: @/
!assignvariableop_7_conv2d_67_bias:@=
/assignvariableop_8_batch_normalization_90_gamma:@<
.assignvariableop_9_batch_normalization_90_beta:@D
6assignvariableop_10_batch_normalization_90_moving_mean:@H
:assignvariableop_11_batch_normalization_90_moving_variance:@?
$assignvariableop_12_conv2d_68_kernel:@�1
"assignvariableop_13_conv2d_68_bias:	�?
0assignvariableop_14_batch_normalization_91_gamma:	�>
/assignvariableop_15_batch_normalization_91_beta:	�E
6assignvariableop_16_batch_normalization_91_moving_mean:	�I
:assignvariableop_17_batch_normalization_91_moving_variance:	�@
$assignvariableop_18_conv2d_69_kernel:��1
"assignvariableop_19_conv2d_69_bias:	�
identity_21��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_66_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_66_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_89_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_89_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_89_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_89_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_67_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_67_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_90_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_90_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_90_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_90_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_68_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_68_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_91_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_91_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_91_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_91_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_69_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_69_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
E__inference_conv2d_66_layer_call_and_return_conditional_losses_174096

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:����������� k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:����������� w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_69_layer_call_fn_174361

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_69_layer_call_and_return_conditional_losses_173380x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������  �: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_174324

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
f
J__inference_leaky_re_lu_87_layer_call_and_return_conditional_losses_173367

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:���������  �*
alpha%���>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�6
�	
C__inference_encoder_layer_call_and_return_conditional_losses_173589

inputs*
conv2d_66_173538: 
conv2d_66_173540: +
batch_normalization_89_173543: +
batch_normalization_89_173545: +
batch_normalization_89_173547: +
batch_normalization_89_173549: *
conv2d_67_173553: @
conv2d_67_173555:@+
batch_normalization_90_173558:@+
batch_normalization_90_173560:@+
batch_normalization_90_173562:@+
batch_normalization_90_173564:@+
conv2d_68_173568:@�
conv2d_68_173570:	�,
batch_normalization_91_173573:	�,
batch_normalization_91_173575:	�,
batch_normalization_91_173577:	�,
batch_normalization_91_173579:	�,
conv2d_69_173583:��
conv2d_69_173585:	�
identity��.batch_normalization_89/StatefulPartitionedCall�.batch_normalization_90/StatefulPartitionedCall�.batch_normalization_91/StatefulPartitionedCall�!conv2d_66/StatefulPartitionedCall�!conv2d_67/StatefulPartitionedCall�!conv2d_68/StatefulPartitionedCall�!conv2d_69/StatefulPartitionedCall�
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_66_173538conv2d_66_173540*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_66_layer_call_and_return_conditional_losses_173281�
.batch_normalization_89/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_89_173543batch_normalization_89_173545batch_normalization_89_173547batch_normalization_89_173549*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_173124�
leaky_re_lu_85/PartitionedCallPartitionedCall7batch_normalization_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_85_layer_call_and_return_conditional_losses_173301�
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_85/PartitionedCall:output:0conv2d_67_173553conv2d_67_173555*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_67_layer_call_and_return_conditional_losses_173314�
.batch_normalization_90/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_90_173558batch_normalization_90_173560batch_normalization_90_173562batch_normalization_90_173564*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_173188�
leaky_re_lu_86/PartitionedCallPartitionedCall7batch_normalization_90/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_86_layer_call_and_return_conditional_losses_173334�
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_86/PartitionedCall:output:0conv2d_68_173568conv2d_68_173570*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_68_layer_call_and_return_conditional_losses_173347�
.batch_normalization_91/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_91_173573batch_normalization_91_173575batch_normalization_91_173577batch_normalization_91_173579*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_173252�
leaky_re_lu_87/PartitionedCallPartitionedCall7batch_normalization_91/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_87_layer_call_and_return_conditional_losses_173367�
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_87/PartitionedCall:output:0conv2d_69_173583conv2d_69_173585*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_69_layer_call_and_return_conditional_losses_173380�
IdentityIdentity*conv2d_69/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp/^batch_normalization_89/StatefulPartitionedCall/^batch_normalization_90/StatefulPartitionedCall/^batch_normalization_91/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:�����������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_89/StatefulPartitionedCall.batch_normalization_89/StatefulPartitionedCall2`
.batch_normalization_90/StatefulPartitionedCall.batch_normalization_90/StatefulPartitionedCall2`
.batch_normalization_91/StatefulPartitionedCall.batch_normalization_91/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
f
J__inference_leaky_re_lu_86_layer_call_and_return_conditional_losses_174260

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������@@@*
alpha%���>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@@:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
�
(__inference_encoder_layer_call_fn_173877

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�&

unknown_17:��

unknown_18:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_173387x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:�����������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_174342

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_90_layer_call_fn_174201

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_173157�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_89_layer_call_fn_174122

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_173124�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_89_layer_call_fn_174109

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_173093�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
(__inference_encoder_layer_call_fn_173677
input_16!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�&

unknown_17:��

unknown_18:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_173589x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:�����������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
input_16
�6
�	
C__inference_encoder_layer_call_and_return_conditional_losses_173731
input_16*
conv2d_66_173680: 
conv2d_66_173682: +
batch_normalization_89_173685: +
batch_normalization_89_173687: +
batch_normalization_89_173689: +
batch_normalization_89_173691: *
conv2d_67_173695: @
conv2d_67_173697:@+
batch_normalization_90_173700:@+
batch_normalization_90_173702:@+
batch_normalization_90_173704:@+
batch_normalization_90_173706:@+
conv2d_68_173710:@�
conv2d_68_173712:	�,
batch_normalization_91_173715:	�,
batch_normalization_91_173717:	�,
batch_normalization_91_173719:	�,
batch_normalization_91_173721:	�,
conv2d_69_173725:��
conv2d_69_173727:	�
identity��.batch_normalization_89/StatefulPartitionedCall�.batch_normalization_90/StatefulPartitionedCall�.batch_normalization_91/StatefulPartitionedCall�!conv2d_66/StatefulPartitionedCall�!conv2d_67/StatefulPartitionedCall�!conv2d_68/StatefulPartitionedCall�!conv2d_69/StatefulPartitionedCall�
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCallinput_16conv2d_66_173680conv2d_66_173682*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_66_layer_call_and_return_conditional_losses_173281�
.batch_normalization_89/StatefulPartitionedCallStatefulPartitionedCall*conv2d_66/StatefulPartitionedCall:output:0batch_normalization_89_173685batch_normalization_89_173687batch_normalization_89_173689batch_normalization_89_173691*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_173093�
leaky_re_lu_85/PartitionedCallPartitionedCall7batch_normalization_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_85_layer_call_and_return_conditional_losses_173301�
!conv2d_67/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_85/PartitionedCall:output:0conv2d_67_173695conv2d_67_173697*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_67_layer_call_and_return_conditional_losses_173314�
.batch_normalization_90/StatefulPartitionedCallStatefulPartitionedCall*conv2d_67/StatefulPartitionedCall:output:0batch_normalization_90_173700batch_normalization_90_173702batch_normalization_90_173704batch_normalization_90_173706*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_173157�
leaky_re_lu_86/PartitionedCallPartitionedCall7batch_normalization_90/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_86_layer_call_and_return_conditional_losses_173334�
!conv2d_68/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_86/PartitionedCall:output:0conv2d_68_173710conv2d_68_173712*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_68_layer_call_and_return_conditional_losses_173347�
.batch_normalization_91/StatefulPartitionedCallStatefulPartitionedCall*conv2d_68/StatefulPartitionedCall:output:0batch_normalization_91_173715batch_normalization_91_173717batch_normalization_91_173719batch_normalization_91_173721*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_173221�
leaky_re_lu_87/PartitionedCallPartitionedCall7batch_normalization_91/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_87_layer_call_and_return_conditional_losses_173367�
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_87/PartitionedCall:output:0conv2d_69_173725conv2d_69_173727*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_69_layer_call_and_return_conditional_losses_173380�
IdentityIdentity*conv2d_69/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp/^batch_normalization_89/StatefulPartitionedCall/^batch_normalization_90/StatefulPartitionedCall/^batch_normalization_91/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall"^conv2d_67/StatefulPartitionedCall"^conv2d_68/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:�����������: : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_89/StatefulPartitionedCall.batch_normalization_89/StatefulPartitionedCall2`
.batch_normalization_90/StatefulPartitionedCall.batch_normalization_90/StatefulPartitionedCall2`
.batch_normalization_91/StatefulPartitionedCall.batch_normalization_91/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2F
!conv2d_67/StatefulPartitionedCall!conv2d_67/StatefulPartitionedCall2F
!conv2d_68/StatefulPartitionedCall!conv2d_68/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
input_16
�u
�
C__inference_encoder_layer_call_and_return_conditional_losses_174076

inputsB
(conv2d_66_conv2d_readvariableop_resource: 7
)conv2d_66_biasadd_readvariableop_resource: <
.batch_normalization_89_readvariableop_resource: >
0batch_normalization_89_readvariableop_1_resource: M
?batch_normalization_89_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_89_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_67_conv2d_readvariableop_resource: @7
)conv2d_67_biasadd_readvariableop_resource:@<
.batch_normalization_90_readvariableop_resource:@>
0batch_normalization_90_readvariableop_1_resource:@M
?batch_normalization_90_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_90_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_68_conv2d_readvariableop_resource:@�8
)conv2d_68_biasadd_readvariableop_resource:	�=
.batch_normalization_91_readvariableop_resource:	�?
0batch_normalization_91_readvariableop_1_resource:	�N
?batch_normalization_91_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_91_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_69_conv2d_readvariableop_resource:��8
)conv2d_69_biasadd_readvariableop_resource:	�
identity��%batch_normalization_89/AssignNewValue�'batch_normalization_89/AssignNewValue_1�6batch_normalization_89/FusedBatchNormV3/ReadVariableOp�8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_89/ReadVariableOp�'batch_normalization_89/ReadVariableOp_1�%batch_normalization_90/AssignNewValue�'batch_normalization_90/AssignNewValue_1�6batch_normalization_90/FusedBatchNormV3/ReadVariableOp�8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_90/ReadVariableOp�'batch_normalization_90/ReadVariableOp_1�%batch_normalization_91/AssignNewValue�'batch_normalization_91/AssignNewValue_1�6batch_normalization_91/FusedBatchNormV3/ReadVariableOp�8batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_91/ReadVariableOp�'batch_normalization_91/ReadVariableOp_1� conv2d_66/BiasAdd/ReadVariableOp�conv2d_66/Conv2D/ReadVariableOp� conv2d_67/BiasAdd/ReadVariableOp�conv2d_67/Conv2D/ReadVariableOp� conv2d_68/BiasAdd/ReadVariableOp�conv2d_68/Conv2D/ReadVariableOp� conv2d_69/BiasAdd/ReadVariableOp�conv2d_69/Conv2D/ReadVariableOp�
conv2d_66/Conv2D/ReadVariableOpReadVariableOp(conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_66/Conv2DConv2Dinputs'conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
 conv2d_66/BiasAdd/ReadVariableOpReadVariableOp)conv2d_66_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_66/BiasAddBiasAddconv2d_66/Conv2D:output:0(conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� n
conv2d_66/ReluReluconv2d_66/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
%batch_normalization_89/ReadVariableOpReadVariableOp.batch_normalization_89_readvariableop_resource*
_output_shapes
: *
dtype0�
'batch_normalization_89/ReadVariableOp_1ReadVariableOp0batch_normalization_89_readvariableop_1_resource*
_output_shapes
: *
dtype0�
6batch_normalization_89/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_89_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_89_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
'batch_normalization_89/FusedBatchNormV3FusedBatchNormV3conv2d_66/Relu:activations:0-batch_normalization_89/ReadVariableOp:value:0/batch_normalization_89/ReadVariableOp_1:value:0>batch_normalization_89/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:����������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_89/AssignNewValueAssignVariableOp?batch_normalization_89_fusedbatchnormv3_readvariableop_resource4batch_normalization_89/FusedBatchNormV3:batch_mean:07^batch_normalization_89/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_89/AssignNewValue_1AssignVariableOpAbatch_normalization_89_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_89/FusedBatchNormV3:batch_variance:09^batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
leaky_re_lu_85/LeakyRelu	LeakyRelu+batch_normalization_89/FusedBatchNormV3:y:0*1
_output_shapes
:����������� *
alpha%���>�
conv2d_67/Conv2D/ReadVariableOpReadVariableOp(conv2d_67_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_67/Conv2DConv2D&leaky_re_lu_85/LeakyRelu:activations:0'conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
 conv2d_67/BiasAdd/ReadVariableOpReadVariableOp)conv2d_67_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_67/BiasAddBiasAddconv2d_67/Conv2D:output:0(conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@l
conv2d_67/ReluReluconv2d_67/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
%batch_normalization_90/ReadVariableOpReadVariableOp.batch_normalization_90_readvariableop_resource*
_output_shapes
:@*
dtype0�
'batch_normalization_90/ReadVariableOp_1ReadVariableOp0batch_normalization_90_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
6batch_normalization_90/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_90_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_90_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
'batch_normalization_90/FusedBatchNormV3FusedBatchNormV3conv2d_67/Relu:activations:0-batch_normalization_90/ReadVariableOp:value:0/batch_normalization_90/ReadVariableOp_1:value:0>batch_normalization_90/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_90/AssignNewValueAssignVariableOp?batch_normalization_90_fusedbatchnormv3_readvariableop_resource4batch_normalization_90/FusedBatchNormV3:batch_mean:07^batch_normalization_90/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_90/AssignNewValue_1AssignVariableOpAbatch_normalization_90_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_90/FusedBatchNormV3:batch_variance:09^batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
leaky_re_lu_86/LeakyRelu	LeakyRelu+batch_normalization_90/FusedBatchNormV3:y:0*/
_output_shapes
:���������@@@*
alpha%���>�
conv2d_68/Conv2D/ReadVariableOpReadVariableOp(conv2d_68_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_68/Conv2DConv2D&leaky_re_lu_86/LeakyRelu:activations:0'conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_68/BiasAdd/ReadVariableOpReadVariableOp)conv2d_68_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_68/BiasAddBiasAddconv2d_68/Conv2D:output:0(conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_68/ReluReluconv2d_68/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
%batch_normalization_91/ReadVariableOpReadVariableOp.batch_normalization_91_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_91/ReadVariableOp_1ReadVariableOp0batch_normalization_91_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_91/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_91_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_91_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_91/FusedBatchNormV3FusedBatchNormV3conv2d_68/Relu:activations:0-batch_normalization_91/ReadVariableOp:value:0/batch_normalization_91/ReadVariableOp_1:value:0>batch_normalization_91/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_91/AssignNewValueAssignVariableOp?batch_normalization_91_fusedbatchnormv3_readvariableop_resource4batch_normalization_91/FusedBatchNormV3:batch_mean:07^batch_normalization_91/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_91/AssignNewValue_1AssignVariableOpAbatch_normalization_91_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_91/FusedBatchNormV3:batch_variance:09^batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
leaky_re_lu_87/LeakyRelu	LeakyRelu+batch_normalization_91/FusedBatchNormV3:y:0*0
_output_shapes
:���������  �*
alpha%���>�
conv2d_69/Conv2D/ReadVariableOpReadVariableOp(conv2d_69_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_69/Conv2DConv2D&leaky_re_lu_87/LeakyRelu:activations:0'conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_69/BiasAdd/ReadVariableOpReadVariableOp)conv2d_69_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_69/BiasAddBiasAddconv2d_69/Conv2D:output:0(conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_69/ReluReluconv2d_69/BiasAdd:output:0*
T0*0
_output_shapes
:����������t
IdentityIdentityconv2d_69/Relu:activations:0^NoOp*
T0*0
_output_shapes
:�����������	
NoOpNoOp&^batch_normalization_89/AssignNewValue(^batch_normalization_89/AssignNewValue_17^batch_normalization_89/FusedBatchNormV3/ReadVariableOp9^batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_89/ReadVariableOp(^batch_normalization_89/ReadVariableOp_1&^batch_normalization_90/AssignNewValue(^batch_normalization_90/AssignNewValue_17^batch_normalization_90/FusedBatchNormV3/ReadVariableOp9^batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_90/ReadVariableOp(^batch_normalization_90/ReadVariableOp_1&^batch_normalization_91/AssignNewValue(^batch_normalization_91/AssignNewValue_17^batch_normalization_91/FusedBatchNormV3/ReadVariableOp9^batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_91/ReadVariableOp(^batch_normalization_91/ReadVariableOp_1!^conv2d_66/BiasAdd/ReadVariableOp ^conv2d_66/Conv2D/ReadVariableOp!^conv2d_67/BiasAdd/ReadVariableOp ^conv2d_67/Conv2D/ReadVariableOp!^conv2d_68/BiasAdd/ReadVariableOp ^conv2d_68/Conv2D/ReadVariableOp!^conv2d_69/BiasAdd/ReadVariableOp ^conv2d_69/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:�����������: : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_89/AssignNewValue%batch_normalization_89/AssignNewValue2R
'batch_normalization_89/AssignNewValue_1'batch_normalization_89/AssignNewValue_12p
6batch_normalization_89/FusedBatchNormV3/ReadVariableOp6batch_normalization_89/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_18batch_normalization_89/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_89/ReadVariableOp%batch_normalization_89/ReadVariableOp2R
'batch_normalization_89/ReadVariableOp_1'batch_normalization_89/ReadVariableOp_12N
%batch_normalization_90/AssignNewValue%batch_normalization_90/AssignNewValue2R
'batch_normalization_90/AssignNewValue_1'batch_normalization_90/AssignNewValue_12p
6batch_normalization_90/FusedBatchNormV3/ReadVariableOp6batch_normalization_90/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_18batch_normalization_90/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_90/ReadVariableOp%batch_normalization_90/ReadVariableOp2R
'batch_normalization_90/ReadVariableOp_1'batch_normalization_90/ReadVariableOp_12N
%batch_normalization_91/AssignNewValue%batch_normalization_91/AssignNewValue2R
'batch_normalization_91/AssignNewValue_1'batch_normalization_91/AssignNewValue_12p
6batch_normalization_91/FusedBatchNormV3/ReadVariableOp6batch_normalization_91/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_91/FusedBatchNormV3/ReadVariableOp_18batch_normalization_91/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_91/ReadVariableOp%batch_normalization_91/ReadVariableOp2R
'batch_normalization_91/ReadVariableOp_1'batch_normalization_91/ReadVariableOp_12D
 conv2d_66/BiasAdd/ReadVariableOp conv2d_66/BiasAdd/ReadVariableOp2B
conv2d_66/Conv2D/ReadVariableOpconv2d_66/Conv2D/ReadVariableOp2D
 conv2d_67/BiasAdd/ReadVariableOp conv2d_67/BiasAdd/ReadVariableOp2B
conv2d_67/Conv2D/ReadVariableOpconv2d_67/Conv2D/ReadVariableOp2D
 conv2d_68/BiasAdd/ReadVariableOp conv2d_68/BiasAdd/ReadVariableOp2B
conv2d_68/Conv2D/ReadVariableOpconv2d_68/Conv2D/ReadVariableOp2D
 conv2d_69/BiasAdd/ReadVariableOp conv2d_69/BiasAdd/ReadVariableOp2B
conv2d_69/Conv2D/ReadVariableOpconv2d_69/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_68_layer_call_and_return_conditional_losses_173347

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������  �j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������  �w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_68_layer_call_and_return_conditional_losses_174280

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������  �j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������  �w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�_
�
C__inference_encoder_layer_call_and_return_conditional_losses_173999

inputsB
(conv2d_66_conv2d_readvariableop_resource: 7
)conv2d_66_biasadd_readvariableop_resource: <
.batch_normalization_89_readvariableop_resource: >
0batch_normalization_89_readvariableop_1_resource: M
?batch_normalization_89_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_89_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_67_conv2d_readvariableop_resource: @7
)conv2d_67_biasadd_readvariableop_resource:@<
.batch_normalization_90_readvariableop_resource:@>
0batch_normalization_90_readvariableop_1_resource:@M
?batch_normalization_90_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_90_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_68_conv2d_readvariableop_resource:@�8
)conv2d_68_biasadd_readvariableop_resource:	�=
.batch_normalization_91_readvariableop_resource:	�?
0batch_normalization_91_readvariableop_1_resource:	�N
?batch_normalization_91_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_91_fusedbatchnormv3_readvariableop_1_resource:	�D
(conv2d_69_conv2d_readvariableop_resource:��8
)conv2d_69_biasadd_readvariableop_resource:	�
identity��6batch_normalization_89/FusedBatchNormV3/ReadVariableOp�8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_89/ReadVariableOp�'batch_normalization_89/ReadVariableOp_1�6batch_normalization_90/FusedBatchNormV3/ReadVariableOp�8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_90/ReadVariableOp�'batch_normalization_90/ReadVariableOp_1�6batch_normalization_91/FusedBatchNormV3/ReadVariableOp�8batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_91/ReadVariableOp�'batch_normalization_91/ReadVariableOp_1� conv2d_66/BiasAdd/ReadVariableOp�conv2d_66/Conv2D/ReadVariableOp� conv2d_67/BiasAdd/ReadVariableOp�conv2d_67/Conv2D/ReadVariableOp� conv2d_68/BiasAdd/ReadVariableOp�conv2d_68/Conv2D/ReadVariableOp� conv2d_69/BiasAdd/ReadVariableOp�conv2d_69/Conv2D/ReadVariableOp�
conv2d_66/Conv2D/ReadVariableOpReadVariableOp(conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_66/Conv2DConv2Dinputs'conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
 conv2d_66/BiasAdd/ReadVariableOpReadVariableOp)conv2d_66_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_66/BiasAddBiasAddconv2d_66/Conv2D:output:0(conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� n
conv2d_66/ReluReluconv2d_66/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
%batch_normalization_89/ReadVariableOpReadVariableOp.batch_normalization_89_readvariableop_resource*
_output_shapes
: *
dtype0�
'batch_normalization_89/ReadVariableOp_1ReadVariableOp0batch_normalization_89_readvariableop_1_resource*
_output_shapes
: *
dtype0�
6batch_normalization_89/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_89_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_89_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
'batch_normalization_89/FusedBatchNormV3FusedBatchNormV3conv2d_66/Relu:activations:0-batch_normalization_89/ReadVariableOp:value:0/batch_normalization_89/ReadVariableOp_1:value:0>batch_normalization_89/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:����������� : : : : :*
epsilon%o�:*
is_training( �
leaky_re_lu_85/LeakyRelu	LeakyRelu+batch_normalization_89/FusedBatchNormV3:y:0*1
_output_shapes
:����������� *
alpha%���>�
conv2d_67/Conv2D/ReadVariableOpReadVariableOp(conv2d_67_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_67/Conv2DConv2D&leaky_re_lu_85/LeakyRelu:activations:0'conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
 conv2d_67/BiasAdd/ReadVariableOpReadVariableOp)conv2d_67_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_67/BiasAddBiasAddconv2d_67/Conv2D:output:0(conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@l
conv2d_67/ReluReluconv2d_67/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
%batch_normalization_90/ReadVariableOpReadVariableOp.batch_normalization_90_readvariableop_resource*
_output_shapes
:@*
dtype0�
'batch_normalization_90/ReadVariableOp_1ReadVariableOp0batch_normalization_90_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
6batch_normalization_90/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_90_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_90_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
'batch_normalization_90/FusedBatchNormV3FusedBatchNormV3conv2d_67/Relu:activations:0-batch_normalization_90/ReadVariableOp:value:0/batch_normalization_90/ReadVariableOp_1:value:0>batch_normalization_90/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@@:@:@:@:@:*
epsilon%o�:*
is_training( �
leaky_re_lu_86/LeakyRelu	LeakyRelu+batch_normalization_90/FusedBatchNormV3:y:0*/
_output_shapes
:���������@@@*
alpha%���>�
conv2d_68/Conv2D/ReadVariableOpReadVariableOp(conv2d_68_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_68/Conv2DConv2D&leaky_re_lu_86/LeakyRelu:activations:0'conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_68/BiasAdd/ReadVariableOpReadVariableOp)conv2d_68_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_68/BiasAddBiasAddconv2d_68/Conv2D:output:0(conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_68/ReluReluconv2d_68/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
%batch_normalization_91/ReadVariableOpReadVariableOp.batch_normalization_91_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_91/ReadVariableOp_1ReadVariableOp0batch_normalization_91_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_91/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_91_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_91_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_91/FusedBatchNormV3FusedBatchNormV3conv2d_68/Relu:activations:0-batch_normalization_91/ReadVariableOp:value:0/batch_normalization_91/ReadVariableOp_1:value:0>batch_normalization_91/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%o�:*
is_training( �
leaky_re_lu_87/LeakyRelu	LeakyRelu+batch_normalization_91/FusedBatchNormV3:y:0*0
_output_shapes
:���������  �*
alpha%���>�
conv2d_69/Conv2D/ReadVariableOpReadVariableOp(conv2d_69_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_69/Conv2DConv2D&leaky_re_lu_87/LeakyRelu:activations:0'conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_69/BiasAdd/ReadVariableOpReadVariableOp)conv2d_69_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_69/BiasAddBiasAddconv2d_69/Conv2D:output:0(conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_69/ReluReluconv2d_69/BiasAdd:output:0*
T0*0
_output_shapes
:����������t
IdentityIdentityconv2d_69/Relu:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp7^batch_normalization_89/FusedBatchNormV3/ReadVariableOp9^batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_89/ReadVariableOp(^batch_normalization_89/ReadVariableOp_17^batch_normalization_90/FusedBatchNormV3/ReadVariableOp9^batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_90/ReadVariableOp(^batch_normalization_90/ReadVariableOp_17^batch_normalization_91/FusedBatchNormV3/ReadVariableOp9^batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_91/ReadVariableOp(^batch_normalization_91/ReadVariableOp_1!^conv2d_66/BiasAdd/ReadVariableOp ^conv2d_66/Conv2D/ReadVariableOp!^conv2d_67/BiasAdd/ReadVariableOp ^conv2d_67/Conv2D/ReadVariableOp!^conv2d_68/BiasAdd/ReadVariableOp ^conv2d_68/Conv2D/ReadVariableOp!^conv2d_69/BiasAdd/ReadVariableOp ^conv2d_69/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:�����������: : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_89/FusedBatchNormV3/ReadVariableOp6batch_normalization_89/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_89/FusedBatchNormV3/ReadVariableOp_18batch_normalization_89/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_89/ReadVariableOp%batch_normalization_89/ReadVariableOp2R
'batch_normalization_89/ReadVariableOp_1'batch_normalization_89/ReadVariableOp_12p
6batch_normalization_90/FusedBatchNormV3/ReadVariableOp6batch_normalization_90/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_90/FusedBatchNormV3/ReadVariableOp_18batch_normalization_90/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_90/ReadVariableOp%batch_normalization_90/ReadVariableOp2R
'batch_normalization_90/ReadVariableOp_1'batch_normalization_90/ReadVariableOp_12p
6batch_normalization_91/FusedBatchNormV3/ReadVariableOp6batch_normalization_91/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_91/FusedBatchNormV3/ReadVariableOp_18batch_normalization_91/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_91/ReadVariableOp%batch_normalization_91/ReadVariableOp2R
'batch_normalization_91/ReadVariableOp_1'batch_normalization_91/ReadVariableOp_12D
 conv2d_66/BiasAdd/ReadVariableOp conv2d_66/BiasAdd/ReadVariableOp2B
conv2d_66/Conv2D/ReadVariableOpconv2d_66/Conv2D/ReadVariableOp2D
 conv2d_67/BiasAdd/ReadVariableOp conv2d_67/BiasAdd/ReadVariableOp2B
conv2d_67/Conv2D/ReadVariableOpconv2d_67/Conv2D/ReadVariableOp2D
 conv2d_68/BiasAdd/ReadVariableOp conv2d_68/BiasAdd/ReadVariableOp2B
conv2d_68/Conv2D/ReadVariableOpconv2d_68/Conv2D/ReadVariableOp2D
 conv2d_69/BiasAdd/ReadVariableOp conv2d_69/BiasAdd/ReadVariableOp2B
conv2d_69/Conv2D/ReadVariableOpconv2d_69/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
K
/__inference_leaky_re_lu_87_layer_call_fn_174347

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_87_layer_call_and_return_conditional_losses_173367i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
K
/__inference_leaky_re_lu_86_layer_call_fn_174255

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_leaky_re_lu_86_layer_call_and_return_conditional_losses_173334h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@@@:W S
/
_output_shapes
:���������@@@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_173093

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
*__inference_conv2d_67_layer_call_fn_174177

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_67_layer_call_and_return_conditional_losses_173314w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
�
*__inference_conv2d_66_layer_call_fn_174085

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_66_layer_call_and_return_conditional_losses_173281y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:����������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
f
J__inference_leaky_re_lu_87_layer_call_and_return_conditional_losses_174352

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:���������  �*
alpha%���>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_90_layer_call_fn_174214

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_173188�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_174158

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_174232

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�k
�
!__inference__wrapped_model_173071
input_16J
0encoder_conv2d_66_conv2d_readvariableop_resource: ?
1encoder_conv2d_66_biasadd_readvariableop_resource: D
6encoder_batch_normalization_89_readvariableop_resource: F
8encoder_batch_normalization_89_readvariableop_1_resource: U
Gencoder_batch_normalization_89_fusedbatchnormv3_readvariableop_resource: W
Iencoder_batch_normalization_89_fusedbatchnormv3_readvariableop_1_resource: J
0encoder_conv2d_67_conv2d_readvariableop_resource: @?
1encoder_conv2d_67_biasadd_readvariableop_resource:@D
6encoder_batch_normalization_90_readvariableop_resource:@F
8encoder_batch_normalization_90_readvariableop_1_resource:@U
Gencoder_batch_normalization_90_fusedbatchnormv3_readvariableop_resource:@W
Iencoder_batch_normalization_90_fusedbatchnormv3_readvariableop_1_resource:@K
0encoder_conv2d_68_conv2d_readvariableop_resource:@�@
1encoder_conv2d_68_biasadd_readvariableop_resource:	�E
6encoder_batch_normalization_91_readvariableop_resource:	�G
8encoder_batch_normalization_91_readvariableop_1_resource:	�V
Gencoder_batch_normalization_91_fusedbatchnormv3_readvariableop_resource:	�X
Iencoder_batch_normalization_91_fusedbatchnormv3_readvariableop_1_resource:	�L
0encoder_conv2d_69_conv2d_readvariableop_resource:��@
1encoder_conv2d_69_biasadd_readvariableop_resource:	�
identity��>encoder/batch_normalization_89/FusedBatchNormV3/ReadVariableOp�@encoder/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1�-encoder/batch_normalization_89/ReadVariableOp�/encoder/batch_normalization_89/ReadVariableOp_1�>encoder/batch_normalization_90/FusedBatchNormV3/ReadVariableOp�@encoder/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1�-encoder/batch_normalization_90/ReadVariableOp�/encoder/batch_normalization_90/ReadVariableOp_1�>encoder/batch_normalization_91/FusedBatchNormV3/ReadVariableOp�@encoder/batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1�-encoder/batch_normalization_91/ReadVariableOp�/encoder/batch_normalization_91/ReadVariableOp_1�(encoder/conv2d_66/BiasAdd/ReadVariableOp�'encoder/conv2d_66/Conv2D/ReadVariableOp�(encoder/conv2d_67/BiasAdd/ReadVariableOp�'encoder/conv2d_67/Conv2D/ReadVariableOp�(encoder/conv2d_68/BiasAdd/ReadVariableOp�'encoder/conv2d_68/Conv2D/ReadVariableOp�(encoder/conv2d_69/BiasAdd/ReadVariableOp�'encoder/conv2d_69/Conv2D/ReadVariableOp�
'encoder/conv2d_66/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
encoder/conv2d_66/Conv2DConv2Dinput_16/encoder/conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
(encoder/conv2d_66/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_66_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
encoder/conv2d_66/BiasAddBiasAdd!encoder/conv2d_66/Conv2D:output:00encoder/conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� ~
encoder/conv2d_66/ReluRelu"encoder/conv2d_66/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
-encoder/batch_normalization_89/ReadVariableOpReadVariableOp6encoder_batch_normalization_89_readvariableop_resource*
_output_shapes
: *
dtype0�
/encoder/batch_normalization_89/ReadVariableOp_1ReadVariableOp8encoder_batch_normalization_89_readvariableop_1_resource*
_output_shapes
: *
dtype0�
>encoder/batch_normalization_89/FusedBatchNormV3/ReadVariableOpReadVariableOpGencoder_batch_normalization_89_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
@encoder/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIencoder_batch_normalization_89_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
/encoder/batch_normalization_89/FusedBatchNormV3FusedBatchNormV3$encoder/conv2d_66/Relu:activations:05encoder/batch_normalization_89/ReadVariableOp:value:07encoder/batch_normalization_89/ReadVariableOp_1:value:0Fencoder/batch_normalization_89/FusedBatchNormV3/ReadVariableOp:value:0Hencoder/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:����������� : : : : :*
epsilon%o�:*
is_training( �
 encoder/leaky_re_lu_85/LeakyRelu	LeakyRelu3encoder/batch_normalization_89/FusedBatchNormV3:y:0*1
_output_shapes
:����������� *
alpha%���>�
'encoder/conv2d_67/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_67_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
encoder/conv2d_67/Conv2DConv2D.encoder/leaky_re_lu_85/LeakyRelu:activations:0/encoder/conv2d_67/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
�
(encoder/conv2d_67/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_67_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
encoder/conv2d_67/BiasAddBiasAdd!encoder/conv2d_67/Conv2D:output:00encoder/conv2d_67/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@|
encoder/conv2d_67/ReluRelu"encoder/conv2d_67/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@@�
-encoder/batch_normalization_90/ReadVariableOpReadVariableOp6encoder_batch_normalization_90_readvariableop_resource*
_output_shapes
:@*
dtype0�
/encoder/batch_normalization_90/ReadVariableOp_1ReadVariableOp8encoder_batch_normalization_90_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
>encoder/batch_normalization_90/FusedBatchNormV3/ReadVariableOpReadVariableOpGencoder_batch_normalization_90_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
@encoder/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIencoder_batch_normalization_90_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
/encoder/batch_normalization_90/FusedBatchNormV3FusedBatchNormV3$encoder/conv2d_67/Relu:activations:05encoder/batch_normalization_90/ReadVariableOp:value:07encoder/batch_normalization_90/ReadVariableOp_1:value:0Fencoder/batch_normalization_90/FusedBatchNormV3/ReadVariableOp:value:0Hencoder/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@@:@:@:@:@:*
epsilon%o�:*
is_training( �
 encoder/leaky_re_lu_86/LeakyRelu	LeakyRelu3encoder/batch_normalization_90/FusedBatchNormV3:y:0*/
_output_shapes
:���������@@@*
alpha%���>�
'encoder/conv2d_68/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_68_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
encoder/conv2d_68/Conv2DConv2D.encoder/leaky_re_lu_86/LeakyRelu:activations:0/encoder/conv2d_68/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
(encoder/conv2d_68/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_68_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
encoder/conv2d_68/BiasAddBiasAdd!encoder/conv2d_68/Conv2D:output:00encoder/conv2d_68/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �}
encoder/conv2d_68/ReluRelu"encoder/conv2d_68/BiasAdd:output:0*
T0*0
_output_shapes
:���������  ��
-encoder/batch_normalization_91/ReadVariableOpReadVariableOp6encoder_batch_normalization_91_readvariableop_resource*
_output_shapes	
:�*
dtype0�
/encoder/batch_normalization_91/ReadVariableOp_1ReadVariableOp8encoder_batch_normalization_91_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
>encoder/batch_normalization_91/FusedBatchNormV3/ReadVariableOpReadVariableOpGencoder_batch_normalization_91_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
@encoder/batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIencoder_batch_normalization_91_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
/encoder/batch_normalization_91/FusedBatchNormV3FusedBatchNormV3$encoder/conv2d_68/Relu:activations:05encoder/batch_normalization_91/ReadVariableOp:value:07encoder/batch_normalization_91/ReadVariableOp_1:value:0Fencoder/batch_normalization_91/FusedBatchNormV3/ReadVariableOp:value:0Hencoder/batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%o�:*
is_training( �
 encoder/leaky_re_lu_87/LeakyRelu	LeakyRelu3encoder/batch_normalization_91/FusedBatchNormV3:y:0*0
_output_shapes
:���������  �*
alpha%���>�
'encoder/conv2d_69/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_69_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
encoder/conv2d_69/Conv2DConv2D.encoder/leaky_re_lu_87/LeakyRelu:activations:0/encoder/conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
(encoder/conv2d_69/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_69_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
encoder/conv2d_69/BiasAddBiasAdd!encoder/conv2d_69/Conv2D:output:00encoder/conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������}
encoder/conv2d_69/ReluRelu"encoder/conv2d_69/BiasAdd:output:0*
T0*0
_output_shapes
:����������|
IdentityIdentity$encoder/conv2d_69/Relu:activations:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp?^encoder/batch_normalization_89/FusedBatchNormV3/ReadVariableOpA^encoder/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1.^encoder/batch_normalization_89/ReadVariableOp0^encoder/batch_normalization_89/ReadVariableOp_1?^encoder/batch_normalization_90/FusedBatchNormV3/ReadVariableOpA^encoder/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1.^encoder/batch_normalization_90/ReadVariableOp0^encoder/batch_normalization_90/ReadVariableOp_1?^encoder/batch_normalization_91/FusedBatchNormV3/ReadVariableOpA^encoder/batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1.^encoder/batch_normalization_91/ReadVariableOp0^encoder/batch_normalization_91/ReadVariableOp_1)^encoder/conv2d_66/BiasAdd/ReadVariableOp(^encoder/conv2d_66/Conv2D/ReadVariableOp)^encoder/conv2d_67/BiasAdd/ReadVariableOp(^encoder/conv2d_67/Conv2D/ReadVariableOp)^encoder/conv2d_68/BiasAdd/ReadVariableOp(^encoder/conv2d_68/Conv2D/ReadVariableOp)^encoder/conv2d_69/BiasAdd/ReadVariableOp(^encoder/conv2d_69/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:�����������: : : : : : : : : : : : : : : : : : : : 2�
>encoder/batch_normalization_89/FusedBatchNormV3/ReadVariableOp>encoder/batch_normalization_89/FusedBatchNormV3/ReadVariableOp2�
@encoder/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_1@encoder/batch_normalization_89/FusedBatchNormV3/ReadVariableOp_12^
-encoder/batch_normalization_89/ReadVariableOp-encoder/batch_normalization_89/ReadVariableOp2b
/encoder/batch_normalization_89/ReadVariableOp_1/encoder/batch_normalization_89/ReadVariableOp_12�
>encoder/batch_normalization_90/FusedBatchNormV3/ReadVariableOp>encoder/batch_normalization_90/FusedBatchNormV3/ReadVariableOp2�
@encoder/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_1@encoder/batch_normalization_90/FusedBatchNormV3/ReadVariableOp_12^
-encoder/batch_normalization_90/ReadVariableOp-encoder/batch_normalization_90/ReadVariableOp2b
/encoder/batch_normalization_90/ReadVariableOp_1/encoder/batch_normalization_90/ReadVariableOp_12�
>encoder/batch_normalization_91/FusedBatchNormV3/ReadVariableOp>encoder/batch_normalization_91/FusedBatchNormV3/ReadVariableOp2�
@encoder/batch_normalization_91/FusedBatchNormV3/ReadVariableOp_1@encoder/batch_normalization_91/FusedBatchNormV3/ReadVariableOp_12^
-encoder/batch_normalization_91/ReadVariableOp-encoder/batch_normalization_91/ReadVariableOp2b
/encoder/batch_normalization_91/ReadVariableOp_1/encoder/batch_normalization_91/ReadVariableOp_12T
(encoder/conv2d_66/BiasAdd/ReadVariableOp(encoder/conv2d_66/BiasAdd/ReadVariableOp2R
'encoder/conv2d_66/Conv2D/ReadVariableOp'encoder/conv2d_66/Conv2D/ReadVariableOp2T
(encoder/conv2d_67/BiasAdd/ReadVariableOp(encoder/conv2d_67/BiasAdd/ReadVariableOp2R
'encoder/conv2d_67/Conv2D/ReadVariableOp'encoder/conv2d_67/Conv2D/ReadVariableOp2T
(encoder/conv2d_68/BiasAdd/ReadVariableOp(encoder/conv2d_68/BiasAdd/ReadVariableOp2R
'encoder/conv2d_68/Conv2D/ReadVariableOp'encoder/conv2d_68/Conv2D/ReadVariableOp2T
(encoder/conv2d_69/BiasAdd/ReadVariableOp(encoder/conv2d_69/BiasAdd/ReadVariableOp2R
'encoder/conv2d_69/Conv2D/ReadVariableOp'encoder/conv2d_69/Conv2D/ReadVariableOp:[ W
1
_output_shapes
:�����������
"
_user_specified_name
input_16
�
�
$__inference_signature_wrapper_173832
input_16!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�&

unknown_17:��

unknown_18:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_173071x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:�����������: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:�����������
"
_user_specified_name
input_16
�
�
E__inference_conv2d_67_layer_call_and_return_conditional_losses_173314

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_91_layer_call_fn_174306

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_173252�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_174250

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_67_layer_call_and_return_conditional_losses_174188

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_173157

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
input_16;
serving_default_input_16:0�����������F
	conv2d_699
StatefulPartitionedCall:0����������tensorflow/serving/predict:�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#axis
	$gamma
%beta
&moving_mean
'moving_variance"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias
 6_jit_compiled_convolution_op"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=axis
	>gamma
?beta
@moving_mean
Amoving_variance"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias
 P_jit_compiled_convolution_op"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias
 j_jit_compiled_convolution_op"
_tf_keras_layer
�
0
1
$2
%3
&4
'5
46
57
>8
?9
@10
A11
N12
O13
X14
Y15
Z16
[17
h18
i19"
trackable_list_wrapper
�
0
1
$2
%3
44
55
>6
?7
N8
O9
X10
Y11
h12
i13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
ptrace_0
qtrace_1
rtrace_2
strace_32�
(__inference_encoder_layer_call_fn_173430
(__inference_encoder_layer_call_fn_173877
(__inference_encoder_layer_call_fn_173922
(__inference_encoder_layer_call_fn_173677�
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
 zptrace_0zqtrace_1zrtrace_2zstrace_3
�
ttrace_0
utrace_1
vtrace_2
wtrace_32�
C__inference_encoder_layer_call_and_return_conditional_losses_173999
C__inference_encoder_layer_call_and_return_conditional_losses_174076
C__inference_encoder_layer_call_and_return_conditional_losses_173731
C__inference_encoder_layer_call_and_return_conditional_losses_173785�
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
 zttrace_0zutrace_1zvtrace_2zwtrace_3
�B�
!__inference__wrapped_model_173071input_16"�
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
xserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
~trace_02�
*__inference_conv2d_66_layer_call_fn_174085�
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
 z~trace_0
�
trace_02�
E__inference_conv2d_66_layer_call_and_return_conditional_losses_174096�
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
 ztrace_0
*:( 2conv2d_66/kernel
: 2conv2d_66/bias
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
<
$0
%1
&2
'3"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_89_layer_call_fn_174109
7__inference_batch_normalization_89_layer_call_fn_174122�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_174140
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_174158�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_89/gamma
):' 2batch_normalization_89/beta
2:0  (2"batch_normalization_89/moving_mean
6:4  (2&batch_normalization_89/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_leaky_re_lu_85_layer_call_fn_174163�
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
 z�trace_0
�
�trace_02�
J__inference_leaky_re_lu_85_layer_call_and_return_conditional_losses_174168�
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
 z�trace_0
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_67_layer_call_fn_174177�
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
 z�trace_0
�
�trace_02�
E__inference_conv2d_67_layer_call_and_return_conditional_losses_174188�
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
 z�trace_0
*:( @2conv2d_67/kernel
:@2conv2d_67/bias
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
<
>0
?1
@2
A3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_90_layer_call_fn_174201
7__inference_batch_normalization_90_layer_call_fn_174214�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_174232
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_174250�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_90/gamma
):'@2batch_normalization_90/beta
2:0@ (2"batch_normalization_90/moving_mean
6:4@ (2&batch_normalization_90/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_leaky_re_lu_86_layer_call_fn_174255�
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
 z�trace_0
�
�trace_02�
J__inference_leaky_re_lu_86_layer_call_and_return_conditional_losses_174260�
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
 z�trace_0
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_68_layer_call_fn_174269�
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
 z�trace_0
�
�trace_02�
E__inference_conv2d_68_layer_call_and_return_conditional_losses_174280�
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
 z�trace_0
+:)@�2conv2d_68/kernel
:�2conv2d_68/bias
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
<
X0
Y1
Z2
[3"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_91_layer_call_fn_174293
7__inference_batch_normalization_91_layer_call_fn_174306�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_174324
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_174342�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
+:)�2batch_normalization_91/gamma
*:(�2batch_normalization_91/beta
3:1� (2"batch_normalization_91/moving_mean
7:5� (2&batch_normalization_91/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_leaky_re_lu_87_layer_call_fn_174347�
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
 z�trace_0
�
�trace_02�
J__inference_leaky_re_lu_87_layer_call_and_return_conditional_losses_174352�
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
 z�trace_0
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_69_layer_call_fn_174361�
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
 z�trace_0
�
�trace_02�
E__inference_conv2d_69_layer_call_and_return_conditional_losses_174372�
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
 z�trace_0
,:*��2conv2d_69/kernel
:�2conv2d_69/bias
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
J
&0
'1
@2
A3
Z4
[5"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_encoder_layer_call_fn_173430input_16"�
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
(__inference_encoder_layer_call_fn_173877inputs"�
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
(__inference_encoder_layer_call_fn_173922inputs"�
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
(__inference_encoder_layer_call_fn_173677input_16"�
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
C__inference_encoder_layer_call_and_return_conditional_losses_173999inputs"�
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
C__inference_encoder_layer_call_and_return_conditional_losses_174076inputs"�
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
C__inference_encoder_layer_call_and_return_conditional_losses_173731input_16"�
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
C__inference_encoder_layer_call_and_return_conditional_losses_173785input_16"�
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
$__inference_signature_wrapper_173832input_16"�
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
*__inference_conv2d_66_layer_call_fn_174085inputs"�
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
�B�
E__inference_conv2d_66_layer_call_and_return_conditional_losses_174096inputs"�
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
.
&0
'1"
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
7__inference_batch_normalization_89_layer_call_fn_174109inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_89_layer_call_fn_174122inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_174140inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_174158inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

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
/__inference_leaky_re_lu_85_layer_call_fn_174163inputs"�
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
�B�
J__inference_leaky_re_lu_85_layer_call_and_return_conditional_losses_174168inputs"�
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
*__inference_conv2d_67_layer_call_fn_174177inputs"�
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
�B�
E__inference_conv2d_67_layer_call_and_return_conditional_losses_174188inputs"�
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
.
@0
A1"
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
7__inference_batch_normalization_90_layer_call_fn_174201inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_90_layer_call_fn_174214inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_174232inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_174250inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

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
/__inference_leaky_re_lu_86_layer_call_fn_174255inputs"�
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
�B�
J__inference_leaky_re_lu_86_layer_call_and_return_conditional_losses_174260inputs"�
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
*__inference_conv2d_68_layer_call_fn_174269inputs"�
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
�B�
E__inference_conv2d_68_layer_call_and_return_conditional_losses_174280inputs"�
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
.
Z0
[1"
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
7__inference_batch_normalization_91_layer_call_fn_174293inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_91_layer_call_fn_174306inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_174324inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_174342inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

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
/__inference_leaky_re_lu_87_layer_call_fn_174347inputs"�
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
�B�
J__inference_leaky_re_lu_87_layer_call_and_return_conditional_losses_174352inputs"�
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
*__inference_conv2d_69_layer_call_fn_174361inputs"�
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
�B�
E__inference_conv2d_69_layer_call_and_return_conditional_losses_174372inputs"�
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
!__inference__wrapped_model_173071�$%&'45>?@ANOXYZ[hi;�8
1�.
,�)
input_16�����������
� ">�;
9
	conv2d_69,�)
	conv2d_69�����������
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_174140�$%&'M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
R__inference_batch_normalization_89_layer_call_and_return_conditional_losses_174158�$%&'M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
7__inference_batch_normalization_89_layer_call_fn_174109�$%&'M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
7__inference_batch_normalization_89_layer_call_fn_174122�$%&'M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_174232�>?@AM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_90_layer_call_and_return_conditional_losses_174250�>?@AM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
7__inference_batch_normalization_90_layer_call_fn_174201�>?@AM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_90_layer_call_fn_174214�>?@AM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_174324�XYZ[N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
R__inference_batch_normalization_91_layer_call_and_return_conditional_losses_174342�XYZ[N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
7__inference_batch_normalization_91_layer_call_fn_174293�XYZ[N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
7__inference_batch_normalization_91_layer_call_fn_174306�XYZ[N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
E__inference_conv2d_66_layer_call_and_return_conditional_losses_174096p9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0����������� 
� �
*__inference_conv2d_66_layer_call_fn_174085c9�6
/�,
*�'
inputs�����������
� ""������������ �
E__inference_conv2d_67_layer_call_and_return_conditional_losses_174188n459�6
/�,
*�'
inputs����������� 
� "-�*
#� 
0���������@@@
� �
*__inference_conv2d_67_layer_call_fn_174177a459�6
/�,
*�'
inputs����������� 
� " ����������@@@�
E__inference_conv2d_68_layer_call_and_return_conditional_losses_174280mNO7�4
-�*
(�%
inputs���������@@@
� ".�+
$�!
0���������  �
� �
*__inference_conv2d_68_layer_call_fn_174269`NO7�4
-�*
(�%
inputs���������@@@
� "!����������  ��
E__inference_conv2d_69_layer_call_and_return_conditional_losses_174372nhi8�5
.�+
)�&
inputs���������  �
� ".�+
$�!
0����������
� �
*__inference_conv2d_69_layer_call_fn_174361ahi8�5
.�+
)�&
inputs���������  �
� "!������������
C__inference_encoder_layer_call_and_return_conditional_losses_173731�$%&'45>?@ANOXYZ[hiC�@
9�6
,�)
input_16�����������
p 

 
� ".�+
$�!
0����������
� �
C__inference_encoder_layer_call_and_return_conditional_losses_173785�$%&'45>?@ANOXYZ[hiC�@
9�6
,�)
input_16�����������
p

 
� ".�+
$�!
0����������
� �
C__inference_encoder_layer_call_and_return_conditional_losses_173999�$%&'45>?@ANOXYZ[hiA�>
7�4
*�'
inputs�����������
p 

 
� ".�+
$�!
0����������
� �
C__inference_encoder_layer_call_and_return_conditional_losses_174076�$%&'45>?@ANOXYZ[hiA�>
7�4
*�'
inputs�����������
p

 
� ".�+
$�!
0����������
� �
(__inference_encoder_layer_call_fn_173430~$%&'45>?@ANOXYZ[hiC�@
9�6
,�)
input_16�����������
p 

 
� "!������������
(__inference_encoder_layer_call_fn_173677~$%&'45>?@ANOXYZ[hiC�@
9�6
,�)
input_16�����������
p

 
� "!������������
(__inference_encoder_layer_call_fn_173877|$%&'45>?@ANOXYZ[hiA�>
7�4
*�'
inputs�����������
p 

 
� "!������������
(__inference_encoder_layer_call_fn_173922|$%&'45>?@ANOXYZ[hiA�>
7�4
*�'
inputs�����������
p

 
� "!������������
J__inference_leaky_re_lu_85_layer_call_and_return_conditional_losses_174168l9�6
/�,
*�'
inputs����������� 
� "/�,
%�"
0����������� 
� �
/__inference_leaky_re_lu_85_layer_call_fn_174163_9�6
/�,
*�'
inputs����������� 
� ""������������ �
J__inference_leaky_re_lu_86_layer_call_and_return_conditional_losses_174260h7�4
-�*
(�%
inputs���������@@@
� "-�*
#� 
0���������@@@
� �
/__inference_leaky_re_lu_86_layer_call_fn_174255[7�4
-�*
(�%
inputs���������@@@
� " ����������@@@�
J__inference_leaky_re_lu_87_layer_call_and_return_conditional_losses_174352j8�5
.�+
)�&
inputs���������  �
� ".�+
$�!
0���������  �
� �
/__inference_leaky_re_lu_87_layer_call_fn_174347]8�5
.�+
)�&
inputs���������  �
� "!����������  ��
$__inference_signature_wrapper_173832�$%&'45>?@ANOXYZ[hiG�D
� 
=�:
8
input_16,�)
input_16�����������">�;
9
	conv2d_69,�)
	conv2d_69����������