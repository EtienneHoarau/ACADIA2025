Йж-
ФЇ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
Р
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
ћ
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
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
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
alphafloat%ЭЬL>"
Ttype0:
2

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02unknown8чУ'

conv2d_transpose_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_32/bias

,conv2d_transpose_32/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_32/bias*
_output_shapes
:*
dtype0

conv2d_transpose_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_32/kernel

.conv2d_transpose_32/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_32/kernel*&
_output_shapes
: *
dtype0
Є
&batch_normalization_59/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_59/moving_variance

:batch_normalization_59/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_59/moving_variance*
_output_shapes
: *
dtype0

"batch_normalization_59/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_59/moving_mean

6batch_normalization_59/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_59/moving_mean*
_output_shapes
: *
dtype0

batch_normalization_59/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_59/beta

/batch_normalization_59/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_59/beta*
_output_shapes
: *
dtype0

batch_normalization_59/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_59/gamma

0batch_normalization_59/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_59/gamma*
_output_shapes
: *
dtype0

conv2d_transpose_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_31/bias

,conv2d_transpose_31/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_31/bias*
_output_shapes
: *
dtype0

conv2d_transpose_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *+
shared_nameconv2d_transpose_31/kernel

.conv2d_transpose_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_31/kernel*&
_output_shapes
:  *
dtype0
Є
&batch_normalization_58/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_58/moving_variance

:batch_normalization_58/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_58/moving_variance*
_output_shapes
: *
dtype0

"batch_normalization_58/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_58/moving_mean

6batch_normalization_58/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_58/moving_mean*
_output_shapes
: *
dtype0

batch_normalization_58/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_58/beta

/batch_normalization_58/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_58/beta*
_output_shapes
: *
dtype0

batch_normalization_58/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_58/gamma

0batch_normalization_58/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_58/gamma*
_output_shapes
: *
dtype0

conv2d_transpose_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_30/bias

,conv2d_transpose_30/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_30/bias*
_output_shapes
: *
dtype0

conv2d_transpose_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_30/kernel

.conv2d_transpose_30/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_30/kernel*&
_output_shapes
: @*
dtype0
Є
&batch_normalization_57/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_57/moving_variance

:batch_normalization_57/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_57/moving_variance*
_output_shapes
:@*
dtype0

"batch_normalization_57/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_57/moving_mean

6batch_normalization_57/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_57/moving_mean*
_output_shapes
:@*
dtype0

batch_normalization_57/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_57/beta

/batch_normalization_57/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_57/beta*
_output_shapes
:@*
dtype0

batch_normalization_57/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_57/gamma

0batch_normalization_57/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_57/gamma*
_output_shapes
:@*
dtype0

conv2d_transpose_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_29/bias

,conv2d_transpose_29/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_29/bias*
_output_shapes
:@*
dtype0

conv2d_transpose_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameconv2d_transpose_29/kernel

.conv2d_transpose_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_29/kernel*'
_output_shapes
:@*
dtype0
Ѕ
&batch_normalization_56/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_56/moving_variance

:batch_normalization_56/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_56/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_56/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_56/moving_mean

6batch_normalization_56/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_56/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_56/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_56/beta

/batch_normalization_56/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_56/beta*
_output_shapes	
:*
dtype0

batch_normalization_56/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_56/gamma

0batch_normalization_56/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_56/gamma*
_output_shapes	
:*
dtype0

conv2d_transpose_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_28/bias

,conv2d_transpose_28/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_28/bias*
_output_shapes	
:*
dtype0

conv2d_transpose_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_28/kernel

.conv2d_transpose_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_28/kernel*'
_output_shapes
: *
dtype0
Є
&batch_normalization_55/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_55/moving_variance

:batch_normalization_55/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_55/moving_variance*
_output_shapes
: *
dtype0

"batch_normalization_55/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_55/moving_mean

6batch_normalization_55/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_55/moving_mean*
_output_shapes
: *
dtype0

batch_normalization_55/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_55/beta

/batch_normalization_55/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_55/beta*
_output_shapes
: *
dtype0

batch_normalization_55/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_55/gamma

0batch_normalization_55/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_55/gamma*
_output_shapes
: *
dtype0

conv2d_transpose_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_27/bias

,conv2d_transpose_27/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_27/bias*
_output_shapes
: *
dtype0

conv2d_transpose_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_27/kernel

.conv2d_transpose_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_27/kernel*&
_output_shapes
: @*
dtype0
Є
&batch_normalization_54/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_54/moving_variance

:batch_normalization_54/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_54/moving_variance*
_output_shapes
:@*
dtype0

"batch_normalization_54/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_54/moving_mean

6batch_normalization_54/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_54/moving_mean*
_output_shapes
:@*
dtype0

batch_normalization_54/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_54/beta

/batch_normalization_54/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_54/beta*
_output_shapes
:@*
dtype0

batch_normalization_54/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_54/gamma

0batch_normalization_54/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_54/gamma*
_output_shapes
:@*
dtype0

conv2d_transpose_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_26/bias

,conv2d_transpose_26/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_26/bias*
_output_shapes
:@*
dtype0

conv2d_transpose_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*+
shared_nameconv2d_transpose_26/kernel

.conv2d_transpose_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_26/kernel*&
_output_shapes
:@@*
dtype0
Є
&batch_normalization_53/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_53/moving_variance

:batch_normalization_53/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_53/moving_variance*
_output_shapes
:@*
dtype0

"batch_normalization_53/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_53/moving_mean

6batch_normalization_53/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_53/moving_mean*
_output_shapes
:@*
dtype0

batch_normalization_53/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_53/beta

/batch_normalization_53/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_53/beta*
_output_shapes
:@*
dtype0

batch_normalization_53/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_53/gamma

0batch_normalization_53/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_53/gamma*
_output_shapes
:@*
dtype0

conv2d_transpose_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_25/bias

,conv2d_transpose_25/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_25/bias*
_output_shapes
:@*
dtype0

conv2d_transpose_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameconv2d_transpose_25/kernel

.conv2d_transpose_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_25/kernel*'
_output_shapes
:@*
dtype0
Ѕ
&batch_normalization_52/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_52/moving_variance

:batch_normalization_52/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_52/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_52/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_52/moving_mean

6batch_normalization_52/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_52/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_52/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_52/beta

/batch_normalization_52/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_52/beta*
_output_shapes	
:*
dtype0

batch_normalization_52/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_52/gamma

0batch_normalization_52/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_52/gamma*
_output_shapes	
:*
dtype0

conv2d_transpose_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_24/bias

,conv2d_transpose_24/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_24/bias*
_output_shapes	
:*
dtype0

conv2d_transpose_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameconv2d_transpose_24/kernel

.conv2d_transpose_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_24/kernel*'
_output_shapes
:@*
dtype0
Є
&batch_normalization_51/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_51/moving_variance

:batch_normalization_51/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_51/moving_variance*
_output_shapes
:@*
dtype0

"batch_normalization_51/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_51/moving_mean

6batch_normalization_51/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_51/moving_mean*
_output_shapes
:@*
dtype0

batch_normalization_51/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_51/beta

/batch_normalization_51/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_51/beta*
_output_shapes
:@*
dtype0

batch_normalization_51/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_51/gamma

0batch_normalization_51/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_51/gamma*
_output_shapes
:@*
dtype0

conv2d_transpose_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_23/bias

,conv2d_transpose_23/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_23/bias*
_output_shapes
:@*
dtype0

conv2d_transpose_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *+
shared_nameconv2d_transpose_23/kernel

.conv2d_transpose_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_23/kernel*&
_output_shapes
:@ *
dtype0
Є
&batch_normalization_50/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_50/moving_variance

:batch_normalization_50/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_50/moving_variance*
_output_shapes
: *
dtype0

"batch_normalization_50/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_50/moving_mean

6batch_normalization_50/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_50/moving_mean*
_output_shapes
: *
dtype0

batch_normalization_50/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_50/beta

/batch_normalization_50/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_50/beta*
_output_shapes
: *
dtype0

batch_normalization_50/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_50/gamma

0batch_normalization_50/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_50/gamma*
_output_shapes
: *
dtype0

conv2d_transpose_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_22/bias

,conv2d_transpose_22/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_22/bias*
_output_shapes
: *
dtype0

conv2d_transpose_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_22/kernel

.conv2d_transpose_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_22/kernel*'
_output_shapes
: *
dtype0

serving_default_input_6Placeholder*0
_output_shapes
:џџџџџџџџџ*
dtype0*%
shape:џџџџџџџџџ
а
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_6conv2d_transpose_22/kernelconv2d_transpose_22/biasbatch_normalization_50/gammabatch_normalization_50/beta"batch_normalization_50/moving_mean&batch_normalization_50/moving_varianceconv2d_transpose_23/kernelconv2d_transpose_23/biasbatch_normalization_51/gammabatch_normalization_51/beta"batch_normalization_51/moving_mean&batch_normalization_51/moving_varianceconv2d_transpose_24/kernelconv2d_transpose_24/biasbatch_normalization_52/gammabatch_normalization_52/beta"batch_normalization_52/moving_mean&batch_normalization_52/moving_varianceconv2d_transpose_25/kernelconv2d_transpose_25/biasbatch_normalization_53/gammabatch_normalization_53/beta"batch_normalization_53/moving_mean&batch_normalization_53/moving_varianceconv2d_transpose_26/kernelconv2d_transpose_26/biasbatch_normalization_54/gammabatch_normalization_54/beta"batch_normalization_54/moving_mean&batch_normalization_54/moving_varianceconv2d_transpose_27/kernelconv2d_transpose_27/biasbatch_normalization_55/gammabatch_normalization_55/beta"batch_normalization_55/moving_mean&batch_normalization_55/moving_varianceconv2d_transpose_28/kernelconv2d_transpose_28/biasbatch_normalization_56/gammabatch_normalization_56/beta"batch_normalization_56/moving_mean&batch_normalization_56/moving_varianceconv2d_transpose_29/kernelconv2d_transpose_29/biasbatch_normalization_57/gammabatch_normalization_57/beta"batch_normalization_57/moving_mean&batch_normalization_57/moving_varianceconv2d_transpose_30/kernelconv2d_transpose_30/biasbatch_normalization_58/gammabatch_normalization_58/beta"batch_normalization_58/moving_mean&batch_normalization_58/moving_varianceconv2d_transpose_31/kernelconv2d_transpose_31/biasbatch_normalization_59/gammabatch_normalization_59/beta"batch_normalization_59/moving_mean&batch_normalization_59/moving_varianceconv2d_transpose_32/kernelconv2d_transpose_32/bias*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_138115

NoOpNoOp
ыс
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ѕс
valueсBс Bс
 	
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
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
layer_with_weights-17
layer-26
layer-27
layer_with_weights-18
layer-28
layer_with_weights-19
layer-29
layer-30
 layer_with_weights-20
 layer-31
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_default_save_signature
(
signatures*
* 
Ш
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op*
е
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8axis
	9gamma
:beta
;moving_mean
<moving_variance*

=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
Ш
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias
 K_jit_compiled_convolution_op*
е
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Raxis
	Sgamma
Tbeta
Umoving_mean
Vmoving_variance*

W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses* 
Ш
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias
 e_jit_compiled_convolution_op*
е
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance*

q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses* 
Ш
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

}kernel
~bias
 _jit_compiled_convolution_op*
р
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
р
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	 axis

Ёgamma
	Ђbeta
Ѓmoving_mean
Єmoving_variance*

Ѕ	variables
Іtrainable_variables
Їregularization_losses
Ј	keras_api
Љ__call__
+Њ&call_and_return_all_conditional_losses* 
б
Ћ	variables
Ќtrainable_variables
­regularization_losses
Ў	keras_api
Џ__call__
+А&call_and_return_all_conditional_losses
Бkernel
	Вbias
!Г_jit_compiled_convolution_op*
р
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses
	Кaxis

Лgamma
	Мbeta
Нmoving_mean
Оmoving_variance*

П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses* 
б
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses
Ыkernel
	Ьbias
!Э_jit_compiled_convolution_op*
р
Ю	variables
Яtrainable_variables
аregularization_losses
б	keras_api
в__call__
+г&call_and_return_all_conditional_losses
	дaxis

еgamma
	жbeta
зmoving_mean
иmoving_variance*

й	variables
кtrainable_variables
лregularization_losses
м	keras_api
н__call__
+о&call_and_return_all_conditional_losses* 
б
п	variables
рtrainable_variables
сregularization_losses
т	keras_api
у__call__
+ф&call_and_return_all_conditional_losses
хkernel
	цbias
!ч_jit_compiled_convolution_op*
р
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses
	юaxis

яgamma
	№beta
ёmoving_mean
ђmoving_variance*

ѓ	variables
єtrainable_variables
ѕregularization_losses
і	keras_api
ї__call__
+ј&call_and_return_all_conditional_losses* 
б
љ	variables
њtrainable_variables
ћregularization_losses
ќ	keras_api
§__call__
+ў&call_and_return_all_conditional_losses
џkernel
	bias
!_jit_compiled_convolution_op*
р
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
б
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
р
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+Ё&call_and_return_all_conditional_losses
	Ђaxis

Ѓgamma
	Єbeta
Ѕmoving_mean
Іmoving_variance*

Ї	variables
Јtrainable_variables
Љregularization_losses
Њ	keras_api
Ћ__call__
+Ќ&call_and_return_all_conditional_losses* 
б
­	variables
Ўtrainable_variables
Џregularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses
Гkernel
	Дbias
!Е_jit_compiled_convolution_op*

/0
01
92
:3
;4
<5
I6
J7
S8
T9
U10
V11
c12
d13
m14
n15
o16
p17
}18
~19
20
21
22
23
24
25
Ё26
Ђ27
Ѓ28
Є29
Б30
В31
Л32
М33
Н34
О35
Ы36
Ь37
е38
ж39
з40
и41
х42
ц43
я44
№45
ё46
ђ47
џ48
49
50
51
52
53
54
55
Ѓ56
Є57
Ѕ58
І59
Г60
Д61*
ц
/0
01
92
:3
I4
J5
S6
T7
c8
d9
m10
n11
}12
~13
14
15
16
17
Ё18
Ђ19
Б20
В21
Л22
М23
Ы24
Ь25
е26
ж27
х28
ц29
я30
№31
џ32
33
34
35
36
37
Ѓ38
Є39
Г40
Д41*
* 
Е
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
'_default_save_signature
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
:
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_3* 
:
Пtrace_0
Рtrace_1
Сtrace_2
Тtrace_3* 
* 

Уserving_default* 

/0
01*

/0
01*
* 

Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

Щtrace_0* 

Ъtrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_22/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_22/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
90
:1
;2
<3*

90
:1*
* 

Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

аtrace_0
бtrace_1* 

вtrace_0
гtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_50/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_50/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_50/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_50/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

йtrace_0* 

кtrace_0* 

I0
J1*

I0
J1*
* 

лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

рtrace_0* 

сtrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_23/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_23/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
S0
T1
U2
V3*

S0
T1*
* 

тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

чtrace_0
шtrace_1* 

щtrace_0
ъtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_51/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_51/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_51/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_51/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses* 

№trace_0* 

ёtrace_0* 

c0
d1*

c0
d1*
* 

ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

їtrace_0* 

јtrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_24/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_24/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
m0
n1
o2
p3*

m0
n1*
* 

љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

ўtrace_0
џtrace_1* 

trace_0
trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_52/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_52/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_52/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_52/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

}0
~1*

}0
~1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*

trace_0* 

trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_25/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_25/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
0
1
2
3*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_53/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_53/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_53/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_53/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

0
1*

0
1*
* 

 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ѕtrace_0* 

Іtrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_26/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_26/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
Ё0
Ђ1
Ѓ2
Є3*

Ё0
Ђ1*
* 

Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ќtrace_0
­trace_1* 

Ўtrace_0
Џtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_54/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_54/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_54/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_54/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses* 

Еtrace_0* 

Жtrace_0* 

Б0
В1*

Б0
В1*
* 

Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
Ћ	variables
Ќtrainable_variables
­regularization_losses
Џ__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses*

Мtrace_0* 

Нtrace_0* 
ke
VARIABLE_VALUEconv2d_transpose_27/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_27/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
Л0
М1
Н2
О3*

Л0
М1*
* 

Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses*

Уtrace_0
Фtrace_1* 

Хtrace_0
Цtrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_55/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_55/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_55/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_55/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses* 

Ьtrace_0* 

Эtrace_0* 

Ы0
Ь1*

Ы0
Ь1*
* 

Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses*

гtrace_0* 

дtrace_0* 
ke
VARIABLE_VALUEconv2d_transpose_28/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_28/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
е0
ж1
з2
и3*

е0
ж1*
* 

еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
Ю	variables
Яtrainable_variables
аregularization_losses
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses*

кtrace_0
лtrace_1* 

мtrace_0
нtrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_56/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_56/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_56/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_56/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

оnon_trainable_variables
пlayers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
й	variables
кtrainable_variables
лregularization_losses
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses* 

уtrace_0* 

фtrace_0* 

х0
ц1*

х0
ц1*
* 

хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
п	variables
рtrainable_variables
сregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses*

ъtrace_0* 

ыtrace_0* 
ke
VARIABLE_VALUEconv2d_transpose_29/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_29/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
я0
№1
ё2
ђ3*

я0
№1*
* 

ьnon_trainable_variables
эlayers
юmetrics
 яlayer_regularization_losses
№layer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses*

ёtrace_0
ђtrace_1* 

ѓtrace_0
єtrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_57/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_57/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_57/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_57/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

ѕnon_trainable_variables
іlayers
їmetrics
 јlayer_regularization_losses
љlayer_metrics
ѓ	variables
єtrainable_variables
ѕregularization_losses
ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses* 

њtrace_0* 

ћtrace_0* 

џ0
1*

џ0
1*
* 

ќnon_trainable_variables
§layers
ўmetrics
 џlayer_regularization_losses
layer_metrics
љ	variables
њtrainable_variables
ћregularization_losses
§__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses*

trace_0* 

trace_0* 
ke
VARIABLE_VALUEconv2d_transpose_30/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_30/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
0
1
2
3*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_58/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_58/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_58/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_58/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
ke
VARIABLE_VALUEconv2d_transpose_31/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_31/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
Ѓ0
Є1
Ѕ2
І3*

Ѓ0
Є1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses*

trace_0
 trace_1* 

Ёtrace_0
Ђtrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_59/gamma6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_59/beta5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_59/moving_mean<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_59/moving_variance@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Ѓnon_trainable_variables
Єlayers
Ѕmetrics
 Іlayer_regularization_losses
Їlayer_metrics
Ї	variables
Јtrainable_variables
Љregularization_losses
Ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses* 

Јtrace_0* 

Љtrace_0* 

Г0
Д1*

Г0
Д1*
* 

Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
­	variables
Ўtrainable_variables
Џregularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses*

Џtrace_0* 

Аtrace_0* 
ke
VARIABLE_VALUEconv2d_transpose_32/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_32/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
Ј
;0
<1
U2
V3
o4
p5
6
7
Ѓ8
Є9
Н10
О11
з12
и13
ё14
ђ15
16
17
Ѕ18
І19*
њ
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
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31*
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
;0
<1*
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
U0
V1*
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
o0
p1*
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

0
1*
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

Ѓ0
Є1*
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

Н0
О1*
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

з0
и1*
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

ё0
ђ1*
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

0
1*
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

Ѕ0
І1*
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
Щ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.conv2d_transpose_22/kernel/Read/ReadVariableOp,conv2d_transpose_22/bias/Read/ReadVariableOp0batch_normalization_50/gamma/Read/ReadVariableOp/batch_normalization_50/beta/Read/ReadVariableOp6batch_normalization_50/moving_mean/Read/ReadVariableOp:batch_normalization_50/moving_variance/Read/ReadVariableOp.conv2d_transpose_23/kernel/Read/ReadVariableOp,conv2d_transpose_23/bias/Read/ReadVariableOp0batch_normalization_51/gamma/Read/ReadVariableOp/batch_normalization_51/beta/Read/ReadVariableOp6batch_normalization_51/moving_mean/Read/ReadVariableOp:batch_normalization_51/moving_variance/Read/ReadVariableOp.conv2d_transpose_24/kernel/Read/ReadVariableOp,conv2d_transpose_24/bias/Read/ReadVariableOp0batch_normalization_52/gamma/Read/ReadVariableOp/batch_normalization_52/beta/Read/ReadVariableOp6batch_normalization_52/moving_mean/Read/ReadVariableOp:batch_normalization_52/moving_variance/Read/ReadVariableOp.conv2d_transpose_25/kernel/Read/ReadVariableOp,conv2d_transpose_25/bias/Read/ReadVariableOp0batch_normalization_53/gamma/Read/ReadVariableOp/batch_normalization_53/beta/Read/ReadVariableOp6batch_normalization_53/moving_mean/Read/ReadVariableOp:batch_normalization_53/moving_variance/Read/ReadVariableOp.conv2d_transpose_26/kernel/Read/ReadVariableOp,conv2d_transpose_26/bias/Read/ReadVariableOp0batch_normalization_54/gamma/Read/ReadVariableOp/batch_normalization_54/beta/Read/ReadVariableOp6batch_normalization_54/moving_mean/Read/ReadVariableOp:batch_normalization_54/moving_variance/Read/ReadVariableOp.conv2d_transpose_27/kernel/Read/ReadVariableOp,conv2d_transpose_27/bias/Read/ReadVariableOp0batch_normalization_55/gamma/Read/ReadVariableOp/batch_normalization_55/beta/Read/ReadVariableOp6batch_normalization_55/moving_mean/Read/ReadVariableOp:batch_normalization_55/moving_variance/Read/ReadVariableOp.conv2d_transpose_28/kernel/Read/ReadVariableOp,conv2d_transpose_28/bias/Read/ReadVariableOp0batch_normalization_56/gamma/Read/ReadVariableOp/batch_normalization_56/beta/Read/ReadVariableOp6batch_normalization_56/moving_mean/Read/ReadVariableOp:batch_normalization_56/moving_variance/Read/ReadVariableOp.conv2d_transpose_29/kernel/Read/ReadVariableOp,conv2d_transpose_29/bias/Read/ReadVariableOp0batch_normalization_57/gamma/Read/ReadVariableOp/batch_normalization_57/beta/Read/ReadVariableOp6batch_normalization_57/moving_mean/Read/ReadVariableOp:batch_normalization_57/moving_variance/Read/ReadVariableOp.conv2d_transpose_30/kernel/Read/ReadVariableOp,conv2d_transpose_30/bias/Read/ReadVariableOp0batch_normalization_58/gamma/Read/ReadVariableOp/batch_normalization_58/beta/Read/ReadVariableOp6batch_normalization_58/moving_mean/Read/ReadVariableOp:batch_normalization_58/moving_variance/Read/ReadVariableOp.conv2d_transpose_31/kernel/Read/ReadVariableOp,conv2d_transpose_31/bias/Read/ReadVariableOp0batch_normalization_59/gamma/Read/ReadVariableOp/batch_normalization_59/beta/Read/ReadVariableOp6batch_normalization_59/moving_mean/Read/ReadVariableOp:batch_normalization_59/moving_variance/Read/ReadVariableOp.conv2d_transpose_32/kernel/Read/ReadVariableOp,conv2d_transpose_32/bias/Read/ReadVariableOpConst*K
TinD
B2@*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_140497
ь
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_transpose_22/kernelconv2d_transpose_22/biasbatch_normalization_50/gammabatch_normalization_50/beta"batch_normalization_50/moving_mean&batch_normalization_50/moving_varianceconv2d_transpose_23/kernelconv2d_transpose_23/biasbatch_normalization_51/gammabatch_normalization_51/beta"batch_normalization_51/moving_mean&batch_normalization_51/moving_varianceconv2d_transpose_24/kernelconv2d_transpose_24/biasbatch_normalization_52/gammabatch_normalization_52/beta"batch_normalization_52/moving_mean&batch_normalization_52/moving_varianceconv2d_transpose_25/kernelconv2d_transpose_25/biasbatch_normalization_53/gammabatch_normalization_53/beta"batch_normalization_53/moving_mean&batch_normalization_53/moving_varianceconv2d_transpose_26/kernelconv2d_transpose_26/biasbatch_normalization_54/gammabatch_normalization_54/beta"batch_normalization_54/moving_mean&batch_normalization_54/moving_varianceconv2d_transpose_27/kernelconv2d_transpose_27/biasbatch_normalization_55/gammabatch_normalization_55/beta"batch_normalization_55/moving_mean&batch_normalization_55/moving_varianceconv2d_transpose_28/kernelconv2d_transpose_28/biasbatch_normalization_56/gammabatch_normalization_56/beta"batch_normalization_56/moving_mean&batch_normalization_56/moving_varianceconv2d_transpose_29/kernelconv2d_transpose_29/biasbatch_normalization_57/gammabatch_normalization_57/beta"batch_normalization_57/moving_mean&batch_normalization_57/moving_varianceconv2d_transpose_30/kernelconv2d_transpose_30/biasbatch_normalization_58/gammabatch_normalization_58/beta"batch_normalization_58/moving_mean&batch_normalization_58/moving_varianceconv2d_transpose_31/kernelconv2d_transpose_31/biasbatch_normalization_59/gammabatch_normalization_59/beta"batch_normalization_59/moving_mean&batch_normalization_59/moving_varianceconv2d_transpose_32/kernelconv2d_transpose_32/bias*J
TinC
A2?*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_140693љ$
Щ
K
/__inference_leaky_re_lu_52_layer_call_fn_139784

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_136841h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

С
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_140007

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
ўћ
ь+
"__inference__traced_restore_140693
file_prefixF
+assignvariableop_conv2d_transpose_22_kernel: 9
+assignvariableop_1_conv2d_transpose_22_bias: =
/assignvariableop_2_batch_normalization_50_gamma: <
.assignvariableop_3_batch_normalization_50_beta: C
5assignvariableop_4_batch_normalization_50_moving_mean: G
9assignvariableop_5_batch_normalization_50_moving_variance: G
-assignvariableop_6_conv2d_transpose_23_kernel:@ 9
+assignvariableop_7_conv2d_transpose_23_bias:@=
/assignvariableop_8_batch_normalization_51_gamma:@<
.assignvariableop_9_batch_normalization_51_beta:@D
6assignvariableop_10_batch_normalization_51_moving_mean:@H
:assignvariableop_11_batch_normalization_51_moving_variance:@I
.assignvariableop_12_conv2d_transpose_24_kernel:@;
,assignvariableop_13_conv2d_transpose_24_bias:	?
0assignvariableop_14_batch_normalization_52_gamma:	>
/assignvariableop_15_batch_normalization_52_beta:	E
6assignvariableop_16_batch_normalization_52_moving_mean:	I
:assignvariableop_17_batch_normalization_52_moving_variance:	I
.assignvariableop_18_conv2d_transpose_25_kernel:@:
,assignvariableop_19_conv2d_transpose_25_bias:@>
0assignvariableop_20_batch_normalization_53_gamma:@=
/assignvariableop_21_batch_normalization_53_beta:@D
6assignvariableop_22_batch_normalization_53_moving_mean:@H
:assignvariableop_23_batch_normalization_53_moving_variance:@H
.assignvariableop_24_conv2d_transpose_26_kernel:@@:
,assignvariableop_25_conv2d_transpose_26_bias:@>
0assignvariableop_26_batch_normalization_54_gamma:@=
/assignvariableop_27_batch_normalization_54_beta:@D
6assignvariableop_28_batch_normalization_54_moving_mean:@H
:assignvariableop_29_batch_normalization_54_moving_variance:@H
.assignvariableop_30_conv2d_transpose_27_kernel: @:
,assignvariableop_31_conv2d_transpose_27_bias: >
0assignvariableop_32_batch_normalization_55_gamma: =
/assignvariableop_33_batch_normalization_55_beta: D
6assignvariableop_34_batch_normalization_55_moving_mean: H
:assignvariableop_35_batch_normalization_55_moving_variance: I
.assignvariableop_36_conv2d_transpose_28_kernel: ;
,assignvariableop_37_conv2d_transpose_28_bias:	?
0assignvariableop_38_batch_normalization_56_gamma:	>
/assignvariableop_39_batch_normalization_56_beta:	E
6assignvariableop_40_batch_normalization_56_moving_mean:	I
:assignvariableop_41_batch_normalization_56_moving_variance:	I
.assignvariableop_42_conv2d_transpose_29_kernel:@:
,assignvariableop_43_conv2d_transpose_29_bias:@>
0assignvariableop_44_batch_normalization_57_gamma:@=
/assignvariableop_45_batch_normalization_57_beta:@D
6assignvariableop_46_batch_normalization_57_moving_mean:@H
:assignvariableop_47_batch_normalization_57_moving_variance:@H
.assignvariableop_48_conv2d_transpose_30_kernel: @:
,assignvariableop_49_conv2d_transpose_30_bias: >
0assignvariableop_50_batch_normalization_58_gamma: =
/assignvariableop_51_batch_normalization_58_beta: D
6assignvariableop_52_batch_normalization_58_moving_mean: H
:assignvariableop_53_batch_normalization_58_moving_variance: H
.assignvariableop_54_conv2d_transpose_31_kernel:  :
,assignvariableop_55_conv2d_transpose_31_bias: >
0assignvariableop_56_batch_normalization_59_gamma: =
/assignvariableop_57_batch_normalization_59_beta: D
6assignvariableop_58_batch_normalization_59_moving_mean: H
:assignvariableop_59_batch_normalization_59_moving_variance: H
.assignvariableop_60_conv2d_transpose_32_kernel: :
,assignvariableop_61_conv2d_transpose_32_bias:
identity_63ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*Л
valueБBЎ?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHё
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*
valueB?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B м
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesџ
ќ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp+assignvariableop_conv2d_transpose_22_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp+assignvariableop_1_conv2d_transpose_22_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_50_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_50_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_50_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_50_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp-assignvariableop_6_conv2d_transpose_23_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp+assignvariableop_7_conv2d_transpose_23_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_51_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_51_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_51_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_51_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp.assignvariableop_12_conv2d_transpose_24_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp,assignvariableop_13_conv2d_transpose_24_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_52_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_52_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_52_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_52_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp.assignvariableop_18_conv2d_transpose_25_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp,assignvariableop_19_conv2d_transpose_25_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_53_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_53_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_53_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_53_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp.assignvariableop_24_conv2d_transpose_26_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp,assignvariableop_25_conv2d_transpose_26_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_54_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_54_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_54_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_54_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp.assignvariableop_30_conv2d_transpose_27_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp,assignvariableop_31_conv2d_transpose_27_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_55_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_55_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_55_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_55_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp.assignvariableop_36_conv2d_transpose_28_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp,assignvariableop_37_conv2d_transpose_28_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_38AssignVariableOp0assignvariableop_38_batch_normalization_56_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_39AssignVariableOp/assignvariableop_39_batch_normalization_56_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_40AssignVariableOp6assignvariableop_40_batch_normalization_56_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_41AssignVariableOp:assignvariableop_41_batch_normalization_56_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp.assignvariableop_42_conv2d_transpose_29_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp,assignvariableop_43_conv2d_transpose_29_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_44AssignVariableOp0assignvariableop_44_batch_normalization_57_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_45AssignVariableOp/assignvariableop_45_batch_normalization_57_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_46AssignVariableOp6assignvariableop_46_batch_normalization_57_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_47AssignVariableOp:assignvariableop_47_batch_normalization_57_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp.assignvariableop_48_conv2d_transpose_30_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp,assignvariableop_49_conv2d_transpose_30_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_50AssignVariableOp0assignvariableop_50_batch_normalization_58_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_51AssignVariableOp/assignvariableop_51_batch_normalization_58_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_52AssignVariableOp6assignvariableop_52_batch_normalization_58_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_53AssignVariableOp:assignvariableop_53_batch_normalization_58_moving_varianceIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp.assignvariableop_54_conv2d_transpose_31_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp,assignvariableop_55_conv2d_transpose_31_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_56AssignVariableOp0assignvariableop_56_batch_normalization_59_gammaIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_57AssignVariableOp/assignvariableop_57_batch_normalization_59_betaIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_58AssignVariableOp6assignvariableop_58_batch_normalization_59_moving_meanIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_59AssignVariableOp:assignvariableop_59_batch_normalization_59_moving_varianceIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOp.assignvariableop_60_conv2d_transpose_32_kernelIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp,assignvariableop_61_conv2d_transpose_32_biasIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ѓ
Identity_62Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_63IdentityIdentity_62:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_63Identity_63:output:0*
_input_shapes
~: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ј
їB
C__inference_decoder_layer_call_and_return_conditional_losses_139101

inputsW
<conv2d_transpose_22_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_22_biasadd_readvariableop_resource: <
.batch_normalization_50_readvariableop_resource: >
0batch_normalization_50_readvariableop_1_resource: M
?batch_normalization_50_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_23_conv2d_transpose_readvariableop_resource:@ A
3conv2d_transpose_23_biasadd_readvariableop_resource:@<
.batch_normalization_51_readvariableop_resource:@>
0batch_normalization_51_readvariableop_1_resource:@M
?batch_normalization_51_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource:@W
<conv2d_transpose_24_conv2d_transpose_readvariableop_resource:@B
3conv2d_transpose_24_biasadd_readvariableop_resource:	=
.batch_normalization_52_readvariableop_resource:	?
0batch_normalization_52_readvariableop_1_resource:	N
?batch_normalization_52_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource:	W
<conv2d_transpose_25_conv2d_transpose_readvariableop_resource:@A
3conv2d_transpose_25_biasadd_readvariableop_resource:@<
.batch_normalization_53_readvariableop_resource:@>
0batch_normalization_53_readvariableop_1_resource:@M
?batch_normalization_53_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_26_conv2d_transpose_readvariableop_resource:@@A
3conv2d_transpose_26_biasadd_readvariableop_resource:@<
.batch_normalization_54_readvariableop_resource:@>
0batch_normalization_54_readvariableop_1_resource:@M
?batch_normalization_54_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_27_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_27_biasadd_readvariableop_resource: <
.batch_normalization_55_readvariableop_resource: >
0batch_normalization_55_readvariableop_1_resource: M
?batch_normalization_55_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource: W
<conv2d_transpose_28_conv2d_transpose_readvariableop_resource: B
3conv2d_transpose_28_biasadd_readvariableop_resource:	=
.batch_normalization_56_readvariableop_resource:	?
0batch_normalization_56_readvariableop_1_resource:	N
?batch_normalization_56_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource:	W
<conv2d_transpose_29_conv2d_transpose_readvariableop_resource:@A
3conv2d_transpose_29_biasadd_readvariableop_resource:@<
.batch_normalization_57_readvariableop_resource:@>
0batch_normalization_57_readvariableop_1_resource:@M
?batch_normalization_57_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_30_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_30_biasadd_readvariableop_resource: <
.batch_normalization_58_readvariableop_resource: >
0batch_normalization_58_readvariableop_1_resource: M
?batch_normalization_58_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_31_conv2d_transpose_readvariableop_resource:  A
3conv2d_transpose_31_biasadd_readvariableop_resource: <
.batch_normalization_59_readvariableop_resource: >
0batch_normalization_59_readvariableop_1_resource: M
?batch_normalization_59_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_59_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_32_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_32_biasadd_readvariableop_resource:
identityЂ%batch_normalization_50/AssignNewValueЂ'batch_normalization_50/AssignNewValue_1Ђ6batch_normalization_50/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_50/ReadVariableOpЂ'batch_normalization_50/ReadVariableOp_1Ђ%batch_normalization_51/AssignNewValueЂ'batch_normalization_51/AssignNewValue_1Ђ6batch_normalization_51/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_51/ReadVariableOpЂ'batch_normalization_51/ReadVariableOp_1Ђ%batch_normalization_52/AssignNewValueЂ'batch_normalization_52/AssignNewValue_1Ђ6batch_normalization_52/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_52/ReadVariableOpЂ'batch_normalization_52/ReadVariableOp_1Ђ%batch_normalization_53/AssignNewValueЂ'batch_normalization_53/AssignNewValue_1Ђ6batch_normalization_53/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_53/ReadVariableOpЂ'batch_normalization_53/ReadVariableOp_1Ђ%batch_normalization_54/AssignNewValueЂ'batch_normalization_54/AssignNewValue_1Ђ6batch_normalization_54/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_54/ReadVariableOpЂ'batch_normalization_54/ReadVariableOp_1Ђ%batch_normalization_55/AssignNewValueЂ'batch_normalization_55/AssignNewValue_1Ђ6batch_normalization_55/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_55/ReadVariableOpЂ'batch_normalization_55/ReadVariableOp_1Ђ%batch_normalization_56/AssignNewValueЂ'batch_normalization_56/AssignNewValue_1Ђ6batch_normalization_56/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_56/ReadVariableOpЂ'batch_normalization_56/ReadVariableOp_1Ђ%batch_normalization_57/AssignNewValueЂ'batch_normalization_57/AssignNewValue_1Ђ6batch_normalization_57/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_57/ReadVariableOpЂ'batch_normalization_57/ReadVariableOp_1Ђ%batch_normalization_58/AssignNewValueЂ'batch_normalization_58/AssignNewValue_1Ђ6batch_normalization_58/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_58/ReadVariableOpЂ'batch_normalization_58/ReadVariableOp_1Ђ%batch_normalization_59/AssignNewValueЂ'batch_normalization_59/AssignNewValue_1Ђ6batch_normalization_59/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_59/ReadVariableOpЂ'batch_normalization_59/ReadVariableOp_1Ђ*conv2d_transpose_22/BiasAdd/ReadVariableOpЂ3conv2d_transpose_22/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_23/BiasAdd/ReadVariableOpЂ3conv2d_transpose_23/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_24/BiasAdd/ReadVariableOpЂ3conv2d_transpose_24/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_25/BiasAdd/ReadVariableOpЂ3conv2d_transpose_25/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_26/BiasAdd/ReadVariableOpЂ3conv2d_transpose_26/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_27/BiasAdd/ReadVariableOpЂ3conv2d_transpose_27/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_28/BiasAdd/ReadVariableOpЂ3conv2d_transpose_28/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_29/BiasAdd/ReadVariableOpЂ3conv2d_transpose_29/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_30/BiasAdd/ReadVariableOpЂ3conv2d_transpose_30/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_31/BiasAdd/ReadVariableOpЂ3conv2d_transpose_31/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_32/BiasAdd/ReadVariableOpЂ3conv2d_transpose_32/conv2d_transpose/ReadVariableOpO
conv2d_transpose_22/ShapeShapeinputs*
T0*
_output_shapes
:q
'conv2d_transpose_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_22/strided_sliceStridedSlice"conv2d_transpose_22/Shape:output:00conv2d_transpose_22/strided_slice/stack:output:02conv2d_transpose_22/strided_slice/stack_1:output:02conv2d_transpose_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_22/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_22/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_22/stack/3Const*
_output_shapes
: *
dtype0*
value	B : э
conv2d_transpose_22/stackPack*conv2d_transpose_22/strided_slice:output:0$conv2d_transpose_22/stack/1:output:0$conv2d_transpose_22/stack/2:output:0$conv2d_transpose_22/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_22/strided_slice_1StridedSlice"conv2d_transpose_22/stack:output:02conv2d_transpose_22/strided_slice_1/stack:output:04conv2d_transpose_22/strided_slice_1/stack_1:output:04conv2d_transpose_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_22/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_22_conv2d_transpose_readvariableop_resource*'
_output_shapes
: *
dtype0
$conv2d_transpose_22/conv2d_transposeConv2DBackpropInput"conv2d_transpose_22/stack:output:0;conv2d_transpose_22/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides

*conv2d_transpose_22/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0У
conv2d_transpose_22/BiasAddBiasAdd-conv2d_transpose_22/conv2d_transpose:output:02conv2d_transpose_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 
%batch_normalization_50/ReadVariableOpReadVariableOp.batch_normalization_50_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_50/ReadVariableOp_1ReadVariableOp0batch_normalization_50_readvariableop_1_resource*
_output_shapes
: *
dtype0В
6batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ж
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0е
'batch_normalization_50/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_22/BiasAdd:output:0-batch_normalization_50/ReadVariableOp:value:0/batch_normalization_50/ReadVariableOp_1:value:0>batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_50/AssignNewValueAssignVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource4batch_normalization_50/FusedBatchNormV3:batch_mean:07^batch_normalization_50/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_50/AssignNewValue_1AssignVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_50/FusedBatchNormV3:batch_variance:09^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
leaky_re_lu_47/LeakyRelu	LeakyRelu+batch_normalization_50/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ *
alpha%>o
conv2d_transpose_23/ShapeShape&leaky_re_lu_47/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_23/strided_sliceStridedSlice"conv2d_transpose_23/Shape:output:00conv2d_transpose_23/strided_slice/stack:output:02conv2d_transpose_23/strided_slice/stack_1:output:02conv2d_transpose_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_23/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_23/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_23/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@э
conv2d_transpose_23/stackPack*conv2d_transpose_23/strided_slice:output:0$conv2d_transpose_23/stack/1:output:0$conv2d_transpose_23/stack/2:output:0$conv2d_transpose_23/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_23/strided_slice_1StridedSlice"conv2d_transpose_23/stack:output:02conv2d_transpose_23/strided_slice_1/stack:output:04conv2d_transpose_23/strided_slice_1/stack_1:output:04conv2d_transpose_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_23/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0І
$conv2d_transpose_23/conv2d_transposeConv2DBackpropInput"conv2d_transpose_23/stack:output:0;conv2d_transpose_23/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_47/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

*conv2d_transpose_23/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0У
conv2d_transpose_23/BiasAddBiasAdd-conv2d_transpose_23/conv2d_transpose:output:02conv2d_transpose_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
%batch_normalization_51/ReadVariableOpReadVariableOp.batch_normalization_51_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_51/ReadVariableOp_1ReadVariableOp0batch_normalization_51_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
6batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0е
'batch_normalization_51/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_23/BiasAdd:output:0-batch_normalization_51/ReadVariableOp:value:0/batch_normalization_51/ReadVariableOp_1:value:0>batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_51/AssignNewValueAssignVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource4batch_normalization_51/FusedBatchNormV3:batch_mean:07^batch_normalization_51/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_51/AssignNewValue_1AssignVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_51/FusedBatchNormV3:batch_variance:09^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
leaky_re_lu_48/LeakyRelu	LeakyRelu+batch_normalization_51/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@*
alpha%>o
conv2d_transpose_24/ShapeShape&leaky_re_lu_48/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_24/strided_sliceStridedSlice"conv2d_transpose_24/Shape:output:00conv2d_transpose_24/strided_slice/stack:output:02conv2d_transpose_24/strided_slice/stack_1:output:02conv2d_transpose_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_24/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_24/stack/2Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_24/stack/3Const*
_output_shapes
: *
dtype0*
value
B :э
conv2d_transpose_24/stackPack*conv2d_transpose_24/strided_slice:output:0$conv2d_transpose_24/stack/1:output:0$conv2d_transpose_24/stack/2:output:0$conv2d_transpose_24/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_24/strided_slice_1StridedSlice"conv2d_transpose_24/stack:output:02conv2d_transpose_24/strided_slice_1/stack:output:04conv2d_transpose_24/strided_slice_1/stack_1:output:04conv2d_transpose_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_24/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_24_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0Ї
$conv2d_transpose_24/conv2d_transposeConv2DBackpropInput"conv2d_transpose_24/stack:output:0;conv2d_transpose_24/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_48/LeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

*conv2d_transpose_24/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ф
conv2d_transpose_24/BiasAddBiasAdd-conv2d_transpose_24/conv2d_transpose:output:02conv2d_transpose_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
%batch_normalization_52/ReadVariableOpReadVariableOp.batch_normalization_52_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_52/ReadVariableOp_1ReadVariableOp0batch_normalization_52_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0к
'batch_normalization_52/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_24/BiasAdd:output:0-batch_normalization_52/ReadVariableOp:value:0/batch_normalization_52/ReadVariableOp_1:value:0>batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_52/AssignNewValueAssignVariableOp?batch_normalization_52_fusedbatchnormv3_readvariableop_resource4batch_normalization_52/FusedBatchNormV3:batch_mean:07^batch_normalization_52/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_52/AssignNewValue_1AssignVariableOpAbatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_52/FusedBatchNormV3:batch_variance:09^batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
leaky_re_lu_49/LeakyRelu	LeakyRelu+batch_normalization_52/FusedBatchNormV3:y:0*0
_output_shapes
:џџџџџџџџџ*
alpha%>o
conv2d_transpose_25/ShapeShape&leaky_re_lu_49/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_25/strided_sliceStridedSlice"conv2d_transpose_25/Shape:output:00conv2d_transpose_25/strided_slice/stack:output:02conv2d_transpose_25/strided_slice/stack_1:output:02conv2d_transpose_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_25/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_25/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_25/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@э
conv2d_transpose_25/stackPack*conv2d_transpose_25/strided_slice:output:0$conv2d_transpose_25/stack/1:output:0$conv2d_transpose_25/stack/2:output:0$conv2d_transpose_25/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_25/strided_slice_1StridedSlice"conv2d_transpose_25/stack:output:02conv2d_transpose_25/strided_slice_1/stack:output:04conv2d_transpose_25/strided_slice_1/stack_1:output:04conv2d_transpose_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_25/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_25_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0І
$conv2d_transpose_25/conv2d_transposeConv2DBackpropInput"conv2d_transpose_25/stack:output:0;conv2d_transpose_25/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_49/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

*conv2d_transpose_25/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0У
conv2d_transpose_25/BiasAddBiasAdd-conv2d_transpose_25/conv2d_transpose:output:02conv2d_transpose_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
%batch_normalization_53/ReadVariableOpReadVariableOp.batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_53/ReadVariableOp_1ReadVariableOp0batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
6batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0е
'batch_normalization_53/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_25/BiasAdd:output:0-batch_normalization_53/ReadVariableOp:value:0/batch_normalization_53/ReadVariableOp_1:value:0>batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_53/AssignNewValueAssignVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource4batch_normalization_53/FusedBatchNormV3:batch_mean:07^batch_normalization_53/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_53/AssignNewValue_1AssignVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_53/FusedBatchNormV3:batch_variance:09^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
leaky_re_lu_50/LeakyRelu	LeakyRelu+batch_normalization_53/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@*
alpha%>o
conv2d_transpose_26/ShapeShape&leaky_re_lu_50/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_26/strided_sliceStridedSlice"conv2d_transpose_26/Shape:output:00conv2d_transpose_26/strided_slice/stack:output:02conv2d_transpose_26/strided_slice/stack_1:output:02conv2d_transpose_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_26/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_26/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_26/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@э
conv2d_transpose_26/stackPack*conv2d_transpose_26/strided_slice:output:0$conv2d_transpose_26/stack/1:output:0$conv2d_transpose_26/stack/2:output:0$conv2d_transpose_26/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_26/strided_slice_1StridedSlice"conv2d_transpose_26/stack:output:02conv2d_transpose_26/strided_slice_1/stack:output:04conv2d_transpose_26/strided_slice_1/stack_1:output:04conv2d_transpose_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_26/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_26_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0І
$conv2d_transpose_26/conv2d_transposeConv2DBackpropInput"conv2d_transpose_26/stack:output:0;conv2d_transpose_26/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_50/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

*conv2d_transpose_26/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0У
conv2d_transpose_26/BiasAddBiasAdd-conv2d_transpose_26/conv2d_transpose:output:02conv2d_transpose_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
%batch_normalization_54/ReadVariableOpReadVariableOp.batch_normalization_54_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_54/ReadVariableOp_1ReadVariableOp0batch_normalization_54_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
6batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0е
'batch_normalization_54/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_26/BiasAdd:output:0-batch_normalization_54/ReadVariableOp:value:0/batch_normalization_54/ReadVariableOp_1:value:0>batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_54/AssignNewValueAssignVariableOp?batch_normalization_54_fusedbatchnormv3_readvariableop_resource4batch_normalization_54/FusedBatchNormV3:batch_mean:07^batch_normalization_54/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_54/AssignNewValue_1AssignVariableOpAbatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_54/FusedBatchNormV3:batch_variance:09^batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
leaky_re_lu_51/LeakyRelu	LeakyRelu+batch_normalization_54/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@*
alpha%>o
conv2d_transpose_27/ShapeShape&leaky_re_lu_51/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_27/strided_sliceStridedSlice"conv2d_transpose_27/Shape:output:00conv2d_transpose_27/strided_slice/stack:output:02conv2d_transpose_27/strided_slice/stack_1:output:02conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value	B : э
conv2d_transpose_27/stackPack*conv2d_transpose_27/strided_slice:output:0$conv2d_transpose_27/stack/1:output:0$conv2d_transpose_27/stack/2:output:0$conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_27/strided_slice_1StridedSlice"conv2d_transpose_27/stack:output:02conv2d_transpose_27/strided_slice_1/stack:output:04conv2d_transpose_27/strided_slice_1/stack_1:output:04conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0І
$conv2d_transpose_27/conv2d_transposeConv2DBackpropInput"conv2d_transpose_27/stack:output:0;conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_51/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

*conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0У
conv2d_transpose_27/BiasAddBiasAdd-conv2d_transpose_27/conv2d_transpose:output:02conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 
%batch_normalization_55/ReadVariableOpReadVariableOp.batch_normalization_55_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_55/ReadVariableOp_1ReadVariableOp0batch_normalization_55_readvariableop_1_resource*
_output_shapes
: *
dtype0В
6batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ж
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0е
'batch_normalization_55/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_27/BiasAdd:output:0-batch_normalization_55/ReadVariableOp:value:0/batch_normalization_55/ReadVariableOp_1:value:0>batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_55/AssignNewValueAssignVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource4batch_normalization_55/FusedBatchNormV3:batch_mean:07^batch_normalization_55/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_55/AssignNewValue_1AssignVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_55/FusedBatchNormV3:batch_variance:09^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
leaky_re_lu_52/LeakyRelu	LeakyRelu+batch_normalization_55/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ *
alpha%>o
conv2d_transpose_28/ShapeShape&leaky_re_lu_52/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_28/strided_sliceStridedSlice"conv2d_transpose_28/Shape:output:00conv2d_transpose_28/strided_slice/stack:output:02conv2d_transpose_28/strided_slice/stack_1:output:02conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ^
conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value
B :э
conv2d_transpose_28/stackPack*conv2d_transpose_28/strided_slice:output:0$conv2d_transpose_28/stack/1:output:0$conv2d_transpose_28/stack/2:output:0$conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_28/strided_slice_1StridedSlice"conv2d_transpose_28/stack:output:02conv2d_transpose_28/strided_slice_1/stack:output:04conv2d_transpose_28/strided_slice_1/stack_1:output:04conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*'
_output_shapes
: *
dtype0Ї
$conv2d_transpose_28/conv2d_transposeConv2DBackpropInput"conv2d_transpose_28/stack:output:0;conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_52/LeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides

*conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ф
conv2d_transpose_28/BiasAddBiasAdd-conv2d_transpose_28/conv2d_transpose:output:02conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  
%batch_normalization_56/ReadVariableOpReadVariableOp.batch_normalization_56_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_56/ReadVariableOp_1ReadVariableOp0batch_normalization_56_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_56/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_56_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0к
'batch_normalization_56/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_28/BiasAdd:output:0-batch_normalization_56/ReadVariableOp:value:0/batch_normalization_56/ReadVariableOp_1:value:0>batch_normalization_56/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ  :::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_56/AssignNewValueAssignVariableOp?batch_normalization_56_fusedbatchnormv3_readvariableop_resource4batch_normalization_56/FusedBatchNormV3:batch_mean:07^batch_normalization_56/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_56/AssignNewValue_1AssignVariableOpAbatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_56/FusedBatchNormV3:batch_variance:09^batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
leaky_re_lu_53/LeakyRelu	LeakyRelu+batch_normalization_56/FusedBatchNormV3:y:0*0
_output_shapes
:џџџџџџџџџ  *
alpha%>o
conv2d_transpose_29/ShapeShape&leaky_re_lu_53/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_29/strided_sliceStridedSlice"conv2d_transpose_29/Shape:output:00conv2d_transpose_29/strided_slice/stack:output:02conv2d_transpose_29/strided_slice/stack_1:output:02conv2d_transpose_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_29/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_29/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_29/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@э
conv2d_transpose_29/stackPack*conv2d_transpose_29/strided_slice:output:0$conv2d_transpose_29/stack/1:output:0$conv2d_transpose_29/stack/2:output:0$conv2d_transpose_29/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_29/strided_slice_1StridedSlice"conv2d_transpose_29/stack:output:02conv2d_transpose_29/strided_slice_1/stack:output:04conv2d_transpose_29/strided_slice_1/stack_1:output:04conv2d_transpose_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_29/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_29_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0І
$conv2d_transpose_29/conv2d_transposeConv2DBackpropInput"conv2d_transpose_29/stack:output:0;conv2d_transpose_29/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_53/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  @*
paddingSAME*
strides

*conv2d_transpose_29/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0У
conv2d_transpose_29/BiasAddBiasAdd-conv2d_transpose_29/conv2d_transpose:output:02conv2d_transpose_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  @
%batch_normalization_57/ReadVariableOpReadVariableOp.batch_normalization_57_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_57/ReadVariableOp_1ReadVariableOp0batch_normalization_57_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
6batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0е
'batch_normalization_57/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_29/BiasAdd:output:0-batch_normalization_57/ReadVariableOp:value:0/batch_normalization_57/ReadVariableOp_1:value:0>batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  @:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_57/AssignNewValueAssignVariableOp?batch_normalization_57_fusedbatchnormv3_readvariableop_resource4batch_normalization_57/FusedBatchNormV3:batch_mean:07^batch_normalization_57/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_57/AssignNewValue_1AssignVariableOpAbatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_57/FusedBatchNormV3:batch_variance:09^batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
leaky_re_lu_54/LeakyRelu	LeakyRelu+batch_normalization_57/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ  @*
alpha%>o
conv2d_transpose_30/ShapeShape&leaky_re_lu_54/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_30/strided_sliceStridedSlice"conv2d_transpose_30/Shape:output:00conv2d_transpose_30/strided_slice/stack:output:02conv2d_transpose_30/strided_slice/stack_1:output:02conv2d_transpose_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_30/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_30/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_30/stack/3Const*
_output_shapes
: *
dtype0*
value	B : э
conv2d_transpose_30/stackPack*conv2d_transpose_30/strided_slice:output:0$conv2d_transpose_30/stack/1:output:0$conv2d_transpose_30/stack/2:output:0$conv2d_transpose_30/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_30/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_30/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_30/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_30/strided_slice_1StridedSlice"conv2d_transpose_30/stack:output:02conv2d_transpose_30/strided_slice_1/stack:output:04conv2d_transpose_30/strided_slice_1/stack_1:output:04conv2d_transpose_30/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_30/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_30_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0І
$conv2d_transpose_30/conv2d_transposeConv2DBackpropInput"conv2d_transpose_30/stack:output:0;conv2d_transpose_30/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_54/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@ *
paddingSAME*
strides

*conv2d_transpose_30/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_30_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0У
conv2d_transpose_30/BiasAddBiasAdd-conv2d_transpose_30/conv2d_transpose:output:02conv2d_transpose_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@ 
%batch_normalization_58/ReadVariableOpReadVariableOp.batch_normalization_58_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_58/ReadVariableOp_1ReadVariableOp0batch_normalization_58_readvariableop_1_resource*
_output_shapes
: *
dtype0В
6batch_normalization_58/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_58_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ж
8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0е
'batch_normalization_58/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_30/BiasAdd:output:0-batch_normalization_58/ReadVariableOp:value:0/batch_normalization_58/ReadVariableOp_1:value:0>batch_normalization_58/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_58/AssignNewValueAssignVariableOp?batch_normalization_58_fusedbatchnormv3_readvariableop_resource4batch_normalization_58/FusedBatchNormV3:batch_mean:07^batch_normalization_58/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_58/AssignNewValue_1AssignVariableOpAbatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_58/FusedBatchNormV3:batch_variance:09^batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
leaky_re_lu_55/LeakyRelu	LeakyRelu+batch_normalization_58/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@@ *
alpha%>o
conv2d_transpose_31/ShapeShape&leaky_re_lu_55/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_31/strided_sliceStridedSlice"conv2d_transpose_31/Shape:output:00conv2d_transpose_31/strided_slice/stack:output:02conv2d_transpose_31/strided_slice/stack_1:output:02conv2d_transpose_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_31/stack/1Const*
_output_shapes
: *
dtype0*
value
B :^
conv2d_transpose_31/stack/2Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_31/stack/3Const*
_output_shapes
: *
dtype0*
value	B : э
conv2d_transpose_31/stackPack*conv2d_transpose_31/strided_slice:output:0$conv2d_transpose_31/stack/1:output:0$conv2d_transpose_31/stack/2:output:0$conv2d_transpose_31/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_31/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_31/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_31/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_31/strided_slice_1StridedSlice"conv2d_transpose_31/stack:output:02conv2d_transpose_31/strided_slice_1/stack:output:04conv2d_transpose_31/strided_slice_1/stack_1:output:04conv2d_transpose_31/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_31/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_31_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0Ј
$conv2d_transpose_31/conv2d_transposeConv2DBackpropInput"conv2d_transpose_31/stack:output:0;conv2d_transpose_31/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_55/LeakyRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

*conv2d_transpose_31/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Х
conv2d_transpose_31/BiasAddBiasAdd-conv2d_transpose_31/conv2d_transpose:output:02conv2d_transpose_31/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ 
%batch_normalization_59/ReadVariableOpReadVariableOp.batch_normalization_59_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_59/ReadVariableOp_1ReadVariableOp0batch_normalization_59_readvariableop_1_resource*
_output_shapes
: *
dtype0В
6batch_normalization_59/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_59_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ж
8batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_59_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0з
'batch_normalization_59/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_31/BiasAdd:output:0-batch_normalization_59/ReadVariableOp:value:0/batch_normalization_59/ReadVariableOp_1:value:0>batch_normalization_59/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_59/AssignNewValueAssignVariableOp?batch_normalization_59_fusedbatchnormv3_readvariableop_resource4batch_normalization_59/FusedBatchNormV3:batch_mean:07^batch_normalization_59/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_59/AssignNewValue_1AssignVariableOpAbatch_normalization_59_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_59/FusedBatchNormV3:batch_variance:09^batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
leaky_re_lu_56/LeakyRelu	LeakyRelu+batch_normalization_59/FusedBatchNormV3:y:0*1
_output_shapes
:џџџџџџџџџ *
alpha%>o
conv2d_transpose_32/ShapeShape&leaky_re_lu_56/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_32/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_32/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_32/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_32/strided_sliceStridedSlice"conv2d_transpose_32/Shape:output:00conv2d_transpose_32/strided_slice/stack:output:02conv2d_transpose_32/strided_slice/stack_1:output:02conv2d_transpose_32/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_32/stack/1Const*
_output_shapes
: *
dtype0*
value
B :^
conv2d_transpose_32/stack/2Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_32/stack/3Const*
_output_shapes
: *
dtype0*
value	B :э
conv2d_transpose_32/stackPack*conv2d_transpose_32/strided_slice:output:0$conv2d_transpose_32/stack/1:output:0$conv2d_transpose_32/stack/2:output:0$conv2d_transpose_32/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_32/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_32/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_32/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_32/strided_slice_1StridedSlice"conv2d_transpose_32/stack:output:02conv2d_transpose_32/strided_slice_1/stack:output:04conv2d_transpose_32/strided_slice_1/stack_1:output:04conv2d_transpose_32/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_32/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_32_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ј
$conv2d_transpose_32/conv2d_transposeConv2DBackpropInput"conv2d_transpose_32/stack:output:0;conv2d_transpose_32/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_56/LeakyRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

*conv2d_transpose_32/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Х
conv2d_transpose_32/BiasAddBiasAdd-conv2d_transpose_32/conv2d_transpose:output:02conv2d_transpose_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ
conv2d_transpose_32/SigmoidSigmoid$conv2d_transpose_32/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџx
IdentityIdentityconv2d_transpose_32/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџї
NoOpNoOp&^batch_normalization_50/AssignNewValue(^batch_normalization_50/AssignNewValue_17^batch_normalization_50/FusedBatchNormV3/ReadVariableOp9^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_50/ReadVariableOp(^batch_normalization_50/ReadVariableOp_1&^batch_normalization_51/AssignNewValue(^batch_normalization_51/AssignNewValue_17^batch_normalization_51/FusedBatchNormV3/ReadVariableOp9^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_51/ReadVariableOp(^batch_normalization_51/ReadVariableOp_1&^batch_normalization_52/AssignNewValue(^batch_normalization_52/AssignNewValue_17^batch_normalization_52/FusedBatchNormV3/ReadVariableOp9^batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_52/ReadVariableOp(^batch_normalization_52/ReadVariableOp_1&^batch_normalization_53/AssignNewValue(^batch_normalization_53/AssignNewValue_17^batch_normalization_53/FusedBatchNormV3/ReadVariableOp9^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_53/ReadVariableOp(^batch_normalization_53/ReadVariableOp_1&^batch_normalization_54/AssignNewValue(^batch_normalization_54/AssignNewValue_17^batch_normalization_54/FusedBatchNormV3/ReadVariableOp9^batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_54/ReadVariableOp(^batch_normalization_54/ReadVariableOp_1&^batch_normalization_55/AssignNewValue(^batch_normalization_55/AssignNewValue_17^batch_normalization_55/FusedBatchNormV3/ReadVariableOp9^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_55/ReadVariableOp(^batch_normalization_55/ReadVariableOp_1&^batch_normalization_56/AssignNewValue(^batch_normalization_56/AssignNewValue_17^batch_normalization_56/FusedBatchNormV3/ReadVariableOp9^batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_56/ReadVariableOp(^batch_normalization_56/ReadVariableOp_1&^batch_normalization_57/AssignNewValue(^batch_normalization_57/AssignNewValue_17^batch_normalization_57/FusedBatchNormV3/ReadVariableOp9^batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_57/ReadVariableOp(^batch_normalization_57/ReadVariableOp_1&^batch_normalization_58/AssignNewValue(^batch_normalization_58/AssignNewValue_17^batch_normalization_58/FusedBatchNormV3/ReadVariableOp9^batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_58/ReadVariableOp(^batch_normalization_58/ReadVariableOp_1&^batch_normalization_59/AssignNewValue(^batch_normalization_59/AssignNewValue_17^batch_normalization_59/FusedBatchNormV3/ReadVariableOp9^batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_59/ReadVariableOp(^batch_normalization_59/ReadVariableOp_1+^conv2d_transpose_22/BiasAdd/ReadVariableOp4^conv2d_transpose_22/conv2d_transpose/ReadVariableOp+^conv2d_transpose_23/BiasAdd/ReadVariableOp4^conv2d_transpose_23/conv2d_transpose/ReadVariableOp+^conv2d_transpose_24/BiasAdd/ReadVariableOp4^conv2d_transpose_24/conv2d_transpose/ReadVariableOp+^conv2d_transpose_25/BiasAdd/ReadVariableOp4^conv2d_transpose_25/conv2d_transpose/ReadVariableOp+^conv2d_transpose_26/BiasAdd/ReadVariableOp4^conv2d_transpose_26/conv2d_transpose/ReadVariableOp+^conv2d_transpose_27/BiasAdd/ReadVariableOp4^conv2d_transpose_27/conv2d_transpose/ReadVariableOp+^conv2d_transpose_28/BiasAdd/ReadVariableOp4^conv2d_transpose_28/conv2d_transpose/ReadVariableOp+^conv2d_transpose_29/BiasAdd/ReadVariableOp4^conv2d_transpose_29/conv2d_transpose/ReadVariableOp+^conv2d_transpose_30/BiasAdd/ReadVariableOp4^conv2d_transpose_30/conv2d_transpose/ReadVariableOp+^conv2d_transpose_31/BiasAdd/ReadVariableOp4^conv2d_transpose_31/conv2d_transpose/ReadVariableOp+^conv2d_transpose_32/BiasAdd/ReadVariableOp4^conv2d_transpose_32/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_50/AssignNewValue%batch_normalization_50/AssignNewValue2R
'batch_normalization_50/AssignNewValue_1'batch_normalization_50/AssignNewValue_12p
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp6batch_normalization_50/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_18batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_50/ReadVariableOp%batch_normalization_50/ReadVariableOp2R
'batch_normalization_50/ReadVariableOp_1'batch_normalization_50/ReadVariableOp_12N
%batch_normalization_51/AssignNewValue%batch_normalization_51/AssignNewValue2R
'batch_normalization_51/AssignNewValue_1'batch_normalization_51/AssignNewValue_12p
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp6batch_normalization_51/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_18batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_51/ReadVariableOp%batch_normalization_51/ReadVariableOp2R
'batch_normalization_51/ReadVariableOp_1'batch_normalization_51/ReadVariableOp_12N
%batch_normalization_52/AssignNewValue%batch_normalization_52/AssignNewValue2R
'batch_normalization_52/AssignNewValue_1'batch_normalization_52/AssignNewValue_12p
6batch_normalization_52/FusedBatchNormV3/ReadVariableOp6batch_normalization_52/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_18batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_52/ReadVariableOp%batch_normalization_52/ReadVariableOp2R
'batch_normalization_52/ReadVariableOp_1'batch_normalization_52/ReadVariableOp_12N
%batch_normalization_53/AssignNewValue%batch_normalization_53/AssignNewValue2R
'batch_normalization_53/AssignNewValue_1'batch_normalization_53/AssignNewValue_12p
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp6batch_normalization_53/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_18batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_53/ReadVariableOp%batch_normalization_53/ReadVariableOp2R
'batch_normalization_53/ReadVariableOp_1'batch_normalization_53/ReadVariableOp_12N
%batch_normalization_54/AssignNewValue%batch_normalization_54/AssignNewValue2R
'batch_normalization_54/AssignNewValue_1'batch_normalization_54/AssignNewValue_12p
6batch_normalization_54/FusedBatchNormV3/ReadVariableOp6batch_normalization_54/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_18batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_54/ReadVariableOp%batch_normalization_54/ReadVariableOp2R
'batch_normalization_54/ReadVariableOp_1'batch_normalization_54/ReadVariableOp_12N
%batch_normalization_55/AssignNewValue%batch_normalization_55/AssignNewValue2R
'batch_normalization_55/AssignNewValue_1'batch_normalization_55/AssignNewValue_12p
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp6batch_normalization_55/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_18batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_55/ReadVariableOp%batch_normalization_55/ReadVariableOp2R
'batch_normalization_55/ReadVariableOp_1'batch_normalization_55/ReadVariableOp_12N
%batch_normalization_56/AssignNewValue%batch_normalization_56/AssignNewValue2R
'batch_normalization_56/AssignNewValue_1'batch_normalization_56/AssignNewValue_12p
6batch_normalization_56/FusedBatchNormV3/ReadVariableOp6batch_normalization_56/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_18batch_normalization_56/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_56/ReadVariableOp%batch_normalization_56/ReadVariableOp2R
'batch_normalization_56/ReadVariableOp_1'batch_normalization_56/ReadVariableOp_12N
%batch_normalization_57/AssignNewValue%batch_normalization_57/AssignNewValue2R
'batch_normalization_57/AssignNewValue_1'batch_normalization_57/AssignNewValue_12p
6batch_normalization_57/FusedBatchNormV3/ReadVariableOp6batch_normalization_57/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_18batch_normalization_57/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_57/ReadVariableOp%batch_normalization_57/ReadVariableOp2R
'batch_normalization_57/ReadVariableOp_1'batch_normalization_57/ReadVariableOp_12N
%batch_normalization_58/AssignNewValue%batch_normalization_58/AssignNewValue2R
'batch_normalization_58/AssignNewValue_1'batch_normalization_58/AssignNewValue_12p
6batch_normalization_58/FusedBatchNormV3/ReadVariableOp6batch_normalization_58/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_18batch_normalization_58/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_58/ReadVariableOp%batch_normalization_58/ReadVariableOp2R
'batch_normalization_58/ReadVariableOp_1'batch_normalization_58/ReadVariableOp_12N
%batch_normalization_59/AssignNewValue%batch_normalization_59/AssignNewValue2R
'batch_normalization_59/AssignNewValue_1'batch_normalization_59/AssignNewValue_12p
6batch_normalization_59/FusedBatchNormV3/ReadVariableOp6batch_normalization_59/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_59/FusedBatchNormV3/ReadVariableOp_18batch_normalization_59/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_59/ReadVariableOp%batch_normalization_59/ReadVariableOp2R
'batch_normalization_59/ReadVariableOp_1'batch_normalization_59/ReadVariableOp_12X
*conv2d_transpose_22/BiasAdd/ReadVariableOp*conv2d_transpose_22/BiasAdd/ReadVariableOp2j
3conv2d_transpose_22/conv2d_transpose/ReadVariableOp3conv2d_transpose_22/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_23/BiasAdd/ReadVariableOp*conv2d_transpose_23/BiasAdd/ReadVariableOp2j
3conv2d_transpose_23/conv2d_transpose/ReadVariableOp3conv2d_transpose_23/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_24/BiasAdd/ReadVariableOp*conv2d_transpose_24/BiasAdd/ReadVariableOp2j
3conv2d_transpose_24/conv2d_transpose/ReadVariableOp3conv2d_transpose_24/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_25/BiasAdd/ReadVariableOp*conv2d_transpose_25/BiasAdd/ReadVariableOp2j
3conv2d_transpose_25/conv2d_transpose/ReadVariableOp3conv2d_transpose_25/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_26/BiasAdd/ReadVariableOp*conv2d_transpose_26/BiasAdd/ReadVariableOp2j
3conv2d_transpose_26/conv2d_transpose/ReadVariableOp3conv2d_transpose_26/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_27/BiasAdd/ReadVariableOp*conv2d_transpose_27/BiasAdd/ReadVariableOp2j
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp3conv2d_transpose_27/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_28/BiasAdd/ReadVariableOp*conv2d_transpose_28/BiasAdd/ReadVariableOp2j
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp3conv2d_transpose_28/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_29/BiasAdd/ReadVariableOp*conv2d_transpose_29/BiasAdd/ReadVariableOp2j
3conv2d_transpose_29/conv2d_transpose/ReadVariableOp3conv2d_transpose_29/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_30/BiasAdd/ReadVariableOp*conv2d_transpose_30/BiasAdd/ReadVariableOp2j
3conv2d_transpose_30/conv2d_transpose/ReadVariableOp3conv2d_transpose_30/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_31/BiasAdd/ReadVariableOp*conv2d_transpose_31/BiasAdd/ReadVariableOp2j
3conv2d_transpose_31/conv2d_transpose/ReadVariableOp3conv2d_transpose_31/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_32/BiasAdd/ReadVariableOp*conv2d_transpose_32/BiasAdd/ReadVariableOp2j
3conv2d_transpose_32/conv2d_transpose/ReadVariableOp3conv2d_transpose_32/conv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
з 

O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_139717

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Ь
Њ
4__inference_conv2d_transpose_29_layer_call_fn_139912

inputs"
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_136378
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_55_layer_call_fn_139743

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_136222
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_136515

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
ЊЊ
ў
C__inference_decoder_layer_call_and_return_conditional_losses_137984
input_65
conv2d_transpose_22_137828: (
conv2d_transpose_22_137830: +
batch_normalization_50_137833: +
batch_normalization_50_137835: +
batch_normalization_50_137837: +
batch_normalization_50_137839: 4
conv2d_transpose_23_137843:@ (
conv2d_transpose_23_137845:@+
batch_normalization_51_137848:@+
batch_normalization_51_137850:@+
batch_normalization_51_137852:@+
batch_normalization_51_137854:@5
conv2d_transpose_24_137858:@)
conv2d_transpose_24_137860:	,
batch_normalization_52_137863:	,
batch_normalization_52_137865:	,
batch_normalization_52_137867:	,
batch_normalization_52_137869:	5
conv2d_transpose_25_137873:@(
conv2d_transpose_25_137875:@+
batch_normalization_53_137878:@+
batch_normalization_53_137880:@+
batch_normalization_53_137882:@+
batch_normalization_53_137884:@4
conv2d_transpose_26_137888:@@(
conv2d_transpose_26_137890:@+
batch_normalization_54_137893:@+
batch_normalization_54_137895:@+
batch_normalization_54_137897:@+
batch_normalization_54_137899:@4
conv2d_transpose_27_137903: @(
conv2d_transpose_27_137905: +
batch_normalization_55_137908: +
batch_normalization_55_137910: +
batch_normalization_55_137912: +
batch_normalization_55_137914: 5
conv2d_transpose_28_137918: )
conv2d_transpose_28_137920:	,
batch_normalization_56_137923:	,
batch_normalization_56_137925:	,
batch_normalization_56_137927:	,
batch_normalization_56_137929:	5
conv2d_transpose_29_137933:@(
conv2d_transpose_29_137935:@+
batch_normalization_57_137938:@+
batch_normalization_57_137940:@+
batch_normalization_57_137942:@+
batch_normalization_57_137944:@4
conv2d_transpose_30_137948: @(
conv2d_transpose_30_137950: +
batch_normalization_58_137953: +
batch_normalization_58_137955: +
batch_normalization_58_137957: +
batch_normalization_58_137959: 4
conv2d_transpose_31_137963:  (
conv2d_transpose_31_137965: +
batch_normalization_59_137968: +
batch_normalization_59_137970: +
batch_normalization_59_137972: +
batch_normalization_59_137974: 4
conv2d_transpose_32_137978: (
conv2d_transpose_32_137980:
identityЂ.batch_normalization_50/StatefulPartitionedCallЂ.batch_normalization_51/StatefulPartitionedCallЂ.batch_normalization_52/StatefulPartitionedCallЂ.batch_normalization_53/StatefulPartitionedCallЂ.batch_normalization_54/StatefulPartitionedCallЂ.batch_normalization_55/StatefulPartitionedCallЂ.batch_normalization_56/StatefulPartitionedCallЂ.batch_normalization_57/StatefulPartitionedCallЂ.batch_normalization_58/StatefulPartitionedCallЂ.batch_normalization_59/StatefulPartitionedCallЂ+conv2d_transpose_22/StatefulPartitionedCallЂ+conv2d_transpose_23/StatefulPartitionedCallЂ+conv2d_transpose_24/StatefulPartitionedCallЂ+conv2d_transpose_25/StatefulPartitionedCallЂ+conv2d_transpose_26/StatefulPartitionedCallЂ+conv2d_transpose_27/StatefulPartitionedCallЂ+conv2d_transpose_28/StatefulPartitionedCallЂ+conv2d_transpose_29/StatefulPartitionedCallЂ+conv2d_transpose_30/StatefulPartitionedCallЂ+conv2d_transpose_31/StatefulPartitionedCallЂ+conv2d_transpose_32/StatefulPartitionedCallЅ
+conv2d_transpose_22/StatefulPartitionedCallStatefulPartitionedCallinput_6conv2d_transpose_22_137828conv2d_transpose_22_137830*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_135622
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_22/StatefulPartitionedCall:output:0batch_normalization_50_137833batch_normalization_50_137835batch_normalization_50_137837batch_normalization_50_137839*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_135682§
leaky_re_lu_47/PartitionedCallPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_47_layer_call_and_return_conditional_losses_136736Х
+conv2d_transpose_23/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_47/PartitionedCall:output:0conv2d_transpose_23_137843conv2d_transpose_23_137845*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_135730
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_23/StatefulPartitionedCall:output:0batch_normalization_51_137848batch_normalization_51_137850batch_normalization_51_137852batch_normalization_51_137854*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_135790§
leaky_re_lu_48/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_48_layer_call_and_return_conditional_losses_136757Ц
+conv2d_transpose_24/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_48/PartitionedCall:output:0conv2d_transpose_24_137858conv2d_transpose_24_137860*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_135838
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_24/StatefulPartitionedCall:output:0batch_normalization_52_137863batch_normalization_52_137865batch_normalization_52_137867batch_normalization_52_137869*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_135898ў
leaky_re_lu_49/PartitionedCallPartitionedCall7batch_normalization_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_136778Х
+conv2d_transpose_25/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_49/PartitionedCall:output:0conv2d_transpose_25_137873conv2d_transpose_25_137875*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_135946
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_25/StatefulPartitionedCall:output:0batch_normalization_53_137878batch_normalization_53_137880batch_normalization_53_137882batch_normalization_53_137884*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_136006§
leaky_re_lu_50/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_136799Х
+conv2d_transpose_26/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_50/PartitionedCall:output:0conv2d_transpose_26_137888conv2d_transpose_26_137890*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_136054
.batch_normalization_54/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_26/StatefulPartitionedCall:output:0batch_normalization_54_137893batch_normalization_54_137895batch_normalization_54_137897batch_normalization_54_137899*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_136114§
leaky_re_lu_51/PartitionedCallPartitionedCall7batch_normalization_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_136820Х
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_51/PartitionedCall:output:0conv2d_transpose_27_137903conv2d_transpose_27_137905*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_136162
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0batch_normalization_55_137908batch_normalization_55_137910batch_normalization_55_137912batch_normalization_55_137914*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_136222§
leaky_re_lu_52/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_136841Ц
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_52/PartitionedCall:output:0conv2d_transpose_28_137918conv2d_transpose_28_137920*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_136270
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0batch_normalization_56_137923batch_normalization_56_137925batch_normalization_56_137927batch_normalization_56_137929*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_136330ў
leaky_re_lu_53/PartitionedCallPartitionedCall7batch_normalization_56/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_136862Х
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_53/PartitionedCall:output:0conv2d_transpose_29_137933conv2d_transpose_29_137935*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_136378
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_29/StatefulPartitionedCall:output:0batch_normalization_57_137938batch_normalization_57_137940batch_normalization_57_137942batch_normalization_57_137944*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_136438§
leaky_re_lu_54/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_136883Х
+conv2d_transpose_30/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_54/PartitionedCall:output:0conv2d_transpose_30_137948conv2d_transpose_30_137950*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_30_layer_call_and_return_conditional_losses_136486
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_30/StatefulPartitionedCall:output:0batch_normalization_58_137953batch_normalization_58_137955batch_normalization_58_137957batch_normalization_58_137959*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_136546§
leaky_re_lu_55/PartitionedCallPartitionedCall7batch_normalization_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_136904Ч
+conv2d_transpose_31/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_55/PartitionedCall:output:0conv2d_transpose_31_137963conv2d_transpose_31_137965*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_31_layer_call_and_return_conditional_losses_136594 
.batch_normalization_59/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_31/StatefulPartitionedCall:output:0batch_normalization_59_137968batch_normalization_59_137970batch_normalization_59_137972batch_normalization_59_137974*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_136654џ
leaky_re_lu_56/PartitionedCallPartitionedCall7batch_normalization_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_136925Ч
+conv2d_transpose_32/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_56/PartitionedCall:output:0conv2d_transpose_32_137978conv2d_transpose_32_137980*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_32_layer_call_and_return_conditional_losses_136703
IdentityIdentity4conv2d_transpose_32/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџЊ
NoOpNoOp/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_52/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_54/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall/^batch_normalization_57/StatefulPartitionedCall/^batch_normalization_58/StatefulPartitionedCall/^batch_normalization_59/StatefulPartitionedCall,^conv2d_transpose_22/StatefulPartitionedCall,^conv2d_transpose_23/StatefulPartitionedCall,^conv2d_transpose_24/StatefulPartitionedCall,^conv2d_transpose_25/StatefulPartitionedCall,^conv2d_transpose_26/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall,^conv2d_transpose_29/StatefulPartitionedCall,^conv2d_transpose_30/StatefulPartitionedCall,^conv2d_transpose_31/StatefulPartitionedCall,^conv2d_transpose_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_54/StatefulPartitionedCall.batch_normalization_54/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2`
.batch_normalization_59/StatefulPartitionedCall.batch_normalization_59/StatefulPartitionedCall2Z
+conv2d_transpose_22/StatefulPartitionedCall+conv2d_transpose_22/StatefulPartitionedCall2Z
+conv2d_transpose_23/StatefulPartitionedCall+conv2d_transpose_23/StatefulPartitionedCall2Z
+conv2d_transpose_24/StatefulPartitionedCall+conv2d_transpose_24/StatefulPartitionedCall2Z
+conv2d_transpose_25/StatefulPartitionedCall+conv2d_transpose_25/StatefulPartitionedCall2Z
+conv2d_transpose_26/StatefulPartitionedCall+conv2d_transpose_26/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2Z
+conv2d_transpose_30/StatefulPartitionedCall+conv2d_transpose_30/StatefulPartitionedCall2Z
+conv2d_transpose_31/StatefulPartitionedCall+conv2d_transpose_31/StatefulPartitionedCall2Z
+conv2d_transpose_32/StatefulPartitionedCall+conv2d_transpose_32/StatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6

f
J__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_136883

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ  @*
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  @:W S
/
_output_shapes
:џџџџџџџџџ  @
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_139903

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:џџџџџџџџџ  *
alpha%>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ  :X T
0
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_135651

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_140245

inputs
identitya
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:џџџџџџџџџ *
alpha%>i
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ :Y U
1
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ф!

O__inference_conv2d_transpose_32_layer_call_and_return_conditional_losses_136703

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџp
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_51_layer_call_fn_139287

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_135790
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Щ
Љ
4__inference_conv2d_transpose_27_layer_call_fn_139684

inputs!
unknown: @
	unknown_0: 
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_136162
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
ОЊ
ў
C__inference_decoder_layer_call_and_return_conditional_losses_137825
input_65
conv2d_transpose_22_137669: (
conv2d_transpose_22_137671: +
batch_normalization_50_137674: +
batch_normalization_50_137676: +
batch_normalization_50_137678: +
batch_normalization_50_137680: 4
conv2d_transpose_23_137684:@ (
conv2d_transpose_23_137686:@+
batch_normalization_51_137689:@+
batch_normalization_51_137691:@+
batch_normalization_51_137693:@+
batch_normalization_51_137695:@5
conv2d_transpose_24_137699:@)
conv2d_transpose_24_137701:	,
batch_normalization_52_137704:	,
batch_normalization_52_137706:	,
batch_normalization_52_137708:	,
batch_normalization_52_137710:	5
conv2d_transpose_25_137714:@(
conv2d_transpose_25_137716:@+
batch_normalization_53_137719:@+
batch_normalization_53_137721:@+
batch_normalization_53_137723:@+
batch_normalization_53_137725:@4
conv2d_transpose_26_137729:@@(
conv2d_transpose_26_137731:@+
batch_normalization_54_137734:@+
batch_normalization_54_137736:@+
batch_normalization_54_137738:@+
batch_normalization_54_137740:@4
conv2d_transpose_27_137744: @(
conv2d_transpose_27_137746: +
batch_normalization_55_137749: +
batch_normalization_55_137751: +
batch_normalization_55_137753: +
batch_normalization_55_137755: 5
conv2d_transpose_28_137759: )
conv2d_transpose_28_137761:	,
batch_normalization_56_137764:	,
batch_normalization_56_137766:	,
batch_normalization_56_137768:	,
batch_normalization_56_137770:	5
conv2d_transpose_29_137774:@(
conv2d_transpose_29_137776:@+
batch_normalization_57_137779:@+
batch_normalization_57_137781:@+
batch_normalization_57_137783:@+
batch_normalization_57_137785:@4
conv2d_transpose_30_137789: @(
conv2d_transpose_30_137791: +
batch_normalization_58_137794: +
batch_normalization_58_137796: +
batch_normalization_58_137798: +
batch_normalization_58_137800: 4
conv2d_transpose_31_137804:  (
conv2d_transpose_31_137806: +
batch_normalization_59_137809: +
batch_normalization_59_137811: +
batch_normalization_59_137813: +
batch_normalization_59_137815: 4
conv2d_transpose_32_137819: (
conv2d_transpose_32_137821:
identityЂ.batch_normalization_50/StatefulPartitionedCallЂ.batch_normalization_51/StatefulPartitionedCallЂ.batch_normalization_52/StatefulPartitionedCallЂ.batch_normalization_53/StatefulPartitionedCallЂ.batch_normalization_54/StatefulPartitionedCallЂ.batch_normalization_55/StatefulPartitionedCallЂ.batch_normalization_56/StatefulPartitionedCallЂ.batch_normalization_57/StatefulPartitionedCallЂ.batch_normalization_58/StatefulPartitionedCallЂ.batch_normalization_59/StatefulPartitionedCallЂ+conv2d_transpose_22/StatefulPartitionedCallЂ+conv2d_transpose_23/StatefulPartitionedCallЂ+conv2d_transpose_24/StatefulPartitionedCallЂ+conv2d_transpose_25/StatefulPartitionedCallЂ+conv2d_transpose_26/StatefulPartitionedCallЂ+conv2d_transpose_27/StatefulPartitionedCallЂ+conv2d_transpose_28/StatefulPartitionedCallЂ+conv2d_transpose_29/StatefulPartitionedCallЂ+conv2d_transpose_30/StatefulPartitionedCallЂ+conv2d_transpose_31/StatefulPartitionedCallЂ+conv2d_transpose_32/StatefulPartitionedCallЅ
+conv2d_transpose_22/StatefulPartitionedCallStatefulPartitionedCallinput_6conv2d_transpose_22_137669conv2d_transpose_22_137671*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_135622 
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_22/StatefulPartitionedCall:output:0batch_normalization_50_137674batch_normalization_50_137676batch_normalization_50_137678batch_normalization_50_137680*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_135651§
leaky_re_lu_47/PartitionedCallPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_47_layer_call_and_return_conditional_losses_136736Х
+conv2d_transpose_23/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_47/PartitionedCall:output:0conv2d_transpose_23_137684conv2d_transpose_23_137686*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_135730 
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_23/StatefulPartitionedCall:output:0batch_normalization_51_137689batch_normalization_51_137691batch_normalization_51_137693batch_normalization_51_137695*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_135759§
leaky_re_lu_48/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_48_layer_call_and_return_conditional_losses_136757Ц
+conv2d_transpose_24/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_48/PartitionedCall:output:0conv2d_transpose_24_137699conv2d_transpose_24_137701*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_135838Ё
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_24/StatefulPartitionedCall:output:0batch_normalization_52_137704batch_normalization_52_137706batch_normalization_52_137708batch_normalization_52_137710*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_135867ў
leaky_re_lu_49/PartitionedCallPartitionedCall7batch_normalization_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_136778Х
+conv2d_transpose_25/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_49/PartitionedCall:output:0conv2d_transpose_25_137714conv2d_transpose_25_137716*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_135946 
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_25/StatefulPartitionedCall:output:0batch_normalization_53_137719batch_normalization_53_137721batch_normalization_53_137723batch_normalization_53_137725*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_135975§
leaky_re_lu_50/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_136799Х
+conv2d_transpose_26/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_50/PartitionedCall:output:0conv2d_transpose_26_137729conv2d_transpose_26_137731*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_136054 
.batch_normalization_54/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_26/StatefulPartitionedCall:output:0batch_normalization_54_137734batch_normalization_54_137736batch_normalization_54_137738batch_normalization_54_137740*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_136083§
leaky_re_lu_51/PartitionedCallPartitionedCall7batch_normalization_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_136820Х
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_51/PartitionedCall:output:0conv2d_transpose_27_137744conv2d_transpose_27_137746*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_136162 
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0batch_normalization_55_137749batch_normalization_55_137751batch_normalization_55_137753batch_normalization_55_137755*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_136191§
leaky_re_lu_52/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_136841Ц
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_52/PartitionedCall:output:0conv2d_transpose_28_137759conv2d_transpose_28_137761*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_136270Ё
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0batch_normalization_56_137764batch_normalization_56_137766batch_normalization_56_137768batch_normalization_56_137770*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_136299ў
leaky_re_lu_53/PartitionedCallPartitionedCall7batch_normalization_56/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_136862Х
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_53/PartitionedCall:output:0conv2d_transpose_29_137774conv2d_transpose_29_137776*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_136378 
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_29/StatefulPartitionedCall:output:0batch_normalization_57_137779batch_normalization_57_137781batch_normalization_57_137783batch_normalization_57_137785*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_136407§
leaky_re_lu_54/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_136883Х
+conv2d_transpose_30/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_54/PartitionedCall:output:0conv2d_transpose_30_137789conv2d_transpose_30_137791*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_30_layer_call_and_return_conditional_losses_136486 
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_30/StatefulPartitionedCall:output:0batch_normalization_58_137794batch_normalization_58_137796batch_normalization_58_137798batch_normalization_58_137800*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_136515§
leaky_re_lu_55/PartitionedCallPartitionedCall7batch_normalization_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_136904Ч
+conv2d_transpose_31/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_55/PartitionedCall:output:0conv2d_transpose_31_137804conv2d_transpose_31_137806*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_31_layer_call_and_return_conditional_losses_136594Ђ
.batch_normalization_59/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_31/StatefulPartitionedCall:output:0batch_normalization_59_137809batch_normalization_59_137811batch_normalization_59_137813batch_normalization_59_137815*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_136623џ
leaky_re_lu_56/PartitionedCallPartitionedCall7batch_normalization_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_136925Ч
+conv2d_transpose_32/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_56/PartitionedCall:output:0conv2d_transpose_32_137819conv2d_transpose_32_137821*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_32_layer_call_and_return_conditional_losses_136703
IdentityIdentity4conv2d_transpose_32/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџЊ
NoOpNoOp/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_52/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_54/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall/^batch_normalization_57/StatefulPartitionedCall/^batch_normalization_58/StatefulPartitionedCall/^batch_normalization_59/StatefulPartitionedCall,^conv2d_transpose_22/StatefulPartitionedCall,^conv2d_transpose_23/StatefulPartitionedCall,^conv2d_transpose_24/StatefulPartitionedCall,^conv2d_transpose_25/StatefulPartitionedCall,^conv2d_transpose_26/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall,^conv2d_transpose_29/StatefulPartitionedCall,^conv2d_transpose_30/StatefulPartitionedCall,^conv2d_transpose_31/StatefulPartitionedCall,^conv2d_transpose_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_54/StatefulPartitionedCall.batch_normalization_54/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2`
.batch_normalization_59/StatefulPartitionedCall.batch_normalization_59/StatefulPartitionedCall2Z
+conv2d_transpose_22/StatefulPartitionedCall+conv2d_transpose_22/StatefulPartitionedCall2Z
+conv2d_transpose_23/StatefulPartitionedCall+conv2d_transpose_23/StatefulPartitionedCall2Z
+conv2d_transpose_24/StatefulPartitionedCall+conv2d_transpose_24/StatefulPartitionedCall2Z
+conv2d_transpose_25/StatefulPartitionedCall+conv2d_transpose_25/StatefulPartitionedCall2Z
+conv2d_transpose_26/StatefulPartitionedCall+conv2d_transpose_26/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2Z
+conv2d_transpose_30/StatefulPartitionedCall+conv2d_transpose_30/StatefulPartitionedCall2Z
+conv2d_transpose_31/StatefulPartitionedCall+conv2d_transpose_31/StatefulPartitionedCall2Z
+conv2d_transpose_32/StatefulPartitionedCall+conv2d_transpose_32/StatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6

f
J__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_140017

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ  @*
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  @:W S
/
_output_shapes
:џџџџџџџџџ  @
 
_user_specified_nameinputs
п 

O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_139375

inputsC
(conv2d_transpose_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :y
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0н
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
ЇЊ
§
C__inference_decoder_layer_call_and_return_conditional_losses_137410

inputs5
conv2d_transpose_22_137254: (
conv2d_transpose_22_137256: +
batch_normalization_50_137259: +
batch_normalization_50_137261: +
batch_normalization_50_137263: +
batch_normalization_50_137265: 4
conv2d_transpose_23_137269:@ (
conv2d_transpose_23_137271:@+
batch_normalization_51_137274:@+
batch_normalization_51_137276:@+
batch_normalization_51_137278:@+
batch_normalization_51_137280:@5
conv2d_transpose_24_137284:@)
conv2d_transpose_24_137286:	,
batch_normalization_52_137289:	,
batch_normalization_52_137291:	,
batch_normalization_52_137293:	,
batch_normalization_52_137295:	5
conv2d_transpose_25_137299:@(
conv2d_transpose_25_137301:@+
batch_normalization_53_137304:@+
batch_normalization_53_137306:@+
batch_normalization_53_137308:@+
batch_normalization_53_137310:@4
conv2d_transpose_26_137314:@@(
conv2d_transpose_26_137316:@+
batch_normalization_54_137319:@+
batch_normalization_54_137321:@+
batch_normalization_54_137323:@+
batch_normalization_54_137325:@4
conv2d_transpose_27_137329: @(
conv2d_transpose_27_137331: +
batch_normalization_55_137334: +
batch_normalization_55_137336: +
batch_normalization_55_137338: +
batch_normalization_55_137340: 5
conv2d_transpose_28_137344: )
conv2d_transpose_28_137346:	,
batch_normalization_56_137349:	,
batch_normalization_56_137351:	,
batch_normalization_56_137353:	,
batch_normalization_56_137355:	5
conv2d_transpose_29_137359:@(
conv2d_transpose_29_137361:@+
batch_normalization_57_137364:@+
batch_normalization_57_137366:@+
batch_normalization_57_137368:@+
batch_normalization_57_137370:@4
conv2d_transpose_30_137374: @(
conv2d_transpose_30_137376: +
batch_normalization_58_137379: +
batch_normalization_58_137381: +
batch_normalization_58_137383: +
batch_normalization_58_137385: 4
conv2d_transpose_31_137389:  (
conv2d_transpose_31_137391: +
batch_normalization_59_137394: +
batch_normalization_59_137396: +
batch_normalization_59_137398: +
batch_normalization_59_137400: 4
conv2d_transpose_32_137404: (
conv2d_transpose_32_137406:
identityЂ.batch_normalization_50/StatefulPartitionedCallЂ.batch_normalization_51/StatefulPartitionedCallЂ.batch_normalization_52/StatefulPartitionedCallЂ.batch_normalization_53/StatefulPartitionedCallЂ.batch_normalization_54/StatefulPartitionedCallЂ.batch_normalization_55/StatefulPartitionedCallЂ.batch_normalization_56/StatefulPartitionedCallЂ.batch_normalization_57/StatefulPartitionedCallЂ.batch_normalization_58/StatefulPartitionedCallЂ.batch_normalization_59/StatefulPartitionedCallЂ+conv2d_transpose_22/StatefulPartitionedCallЂ+conv2d_transpose_23/StatefulPartitionedCallЂ+conv2d_transpose_24/StatefulPartitionedCallЂ+conv2d_transpose_25/StatefulPartitionedCallЂ+conv2d_transpose_26/StatefulPartitionedCallЂ+conv2d_transpose_27/StatefulPartitionedCallЂ+conv2d_transpose_28/StatefulPartitionedCallЂ+conv2d_transpose_29/StatefulPartitionedCallЂ+conv2d_transpose_30/StatefulPartitionedCallЂ+conv2d_transpose_31/StatefulPartitionedCallЂ+conv2d_transpose_32/StatefulPartitionedCallЄ
+conv2d_transpose_22/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_22_137254conv2d_transpose_22_137256*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_135622
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_22/StatefulPartitionedCall:output:0batch_normalization_50_137259batch_normalization_50_137261batch_normalization_50_137263batch_normalization_50_137265*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_135682§
leaky_re_lu_47/PartitionedCallPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_47_layer_call_and_return_conditional_losses_136736Х
+conv2d_transpose_23/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_47/PartitionedCall:output:0conv2d_transpose_23_137269conv2d_transpose_23_137271*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_135730
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_23/StatefulPartitionedCall:output:0batch_normalization_51_137274batch_normalization_51_137276batch_normalization_51_137278batch_normalization_51_137280*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_135790§
leaky_re_lu_48/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_48_layer_call_and_return_conditional_losses_136757Ц
+conv2d_transpose_24/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_48/PartitionedCall:output:0conv2d_transpose_24_137284conv2d_transpose_24_137286*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_135838
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_24/StatefulPartitionedCall:output:0batch_normalization_52_137289batch_normalization_52_137291batch_normalization_52_137293batch_normalization_52_137295*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_135898ў
leaky_re_lu_49/PartitionedCallPartitionedCall7batch_normalization_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_136778Х
+conv2d_transpose_25/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_49/PartitionedCall:output:0conv2d_transpose_25_137299conv2d_transpose_25_137301*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_135946
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_25/StatefulPartitionedCall:output:0batch_normalization_53_137304batch_normalization_53_137306batch_normalization_53_137308batch_normalization_53_137310*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_136006§
leaky_re_lu_50/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_136799Х
+conv2d_transpose_26/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_50/PartitionedCall:output:0conv2d_transpose_26_137314conv2d_transpose_26_137316*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_136054
.batch_normalization_54/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_26/StatefulPartitionedCall:output:0batch_normalization_54_137319batch_normalization_54_137321batch_normalization_54_137323batch_normalization_54_137325*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_136114§
leaky_re_lu_51/PartitionedCallPartitionedCall7batch_normalization_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_136820Х
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_51/PartitionedCall:output:0conv2d_transpose_27_137329conv2d_transpose_27_137331*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_136162
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0batch_normalization_55_137334batch_normalization_55_137336batch_normalization_55_137338batch_normalization_55_137340*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_136222§
leaky_re_lu_52/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_136841Ц
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_52/PartitionedCall:output:0conv2d_transpose_28_137344conv2d_transpose_28_137346*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_136270
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0batch_normalization_56_137349batch_normalization_56_137351batch_normalization_56_137353batch_normalization_56_137355*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_136330ў
leaky_re_lu_53/PartitionedCallPartitionedCall7batch_normalization_56/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_136862Х
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_53/PartitionedCall:output:0conv2d_transpose_29_137359conv2d_transpose_29_137361*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_136378
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_29/StatefulPartitionedCall:output:0batch_normalization_57_137364batch_normalization_57_137366batch_normalization_57_137368batch_normalization_57_137370*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_136438§
leaky_re_lu_54/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_136883Х
+conv2d_transpose_30/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_54/PartitionedCall:output:0conv2d_transpose_30_137374conv2d_transpose_30_137376*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_30_layer_call_and_return_conditional_losses_136486
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_30/StatefulPartitionedCall:output:0batch_normalization_58_137379batch_normalization_58_137381batch_normalization_58_137383batch_normalization_58_137385*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_136546§
leaky_re_lu_55/PartitionedCallPartitionedCall7batch_normalization_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_136904Ч
+conv2d_transpose_31/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_55/PartitionedCall:output:0conv2d_transpose_31_137389conv2d_transpose_31_137391*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_31_layer_call_and_return_conditional_losses_136594 
.batch_normalization_59/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_31/StatefulPartitionedCall:output:0batch_normalization_59_137394batch_normalization_59_137396batch_normalization_59_137398batch_normalization_59_137400*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_136654џ
leaky_re_lu_56/PartitionedCallPartitionedCall7batch_normalization_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_136925Ч
+conv2d_transpose_32/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_56/PartitionedCall:output:0conv2d_transpose_32_137404conv2d_transpose_32_137406*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_32_layer_call_and_return_conditional_losses_136703
IdentityIdentity4conv2d_transpose_32/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџЊ
NoOpNoOp/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_52/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_54/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall/^batch_normalization_57/StatefulPartitionedCall/^batch_normalization_58/StatefulPartitionedCall/^batch_normalization_59/StatefulPartitionedCall,^conv2d_transpose_22/StatefulPartitionedCall,^conv2d_transpose_23/StatefulPartitionedCall,^conv2d_transpose_24/StatefulPartitionedCall,^conv2d_transpose_25/StatefulPartitionedCall,^conv2d_transpose_26/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall,^conv2d_transpose_29/StatefulPartitionedCall,^conv2d_transpose_30/StatefulPartitionedCall,^conv2d_transpose_31/StatefulPartitionedCall,^conv2d_transpose_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_54/StatefulPartitionedCall.batch_normalization_54/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2`
.batch_normalization_59/StatefulPartitionedCall.batch_normalization_59/StatefulPartitionedCall2Z
+conv2d_transpose_22/StatefulPartitionedCall+conv2d_transpose_22/StatefulPartitionedCall2Z
+conv2d_transpose_23/StatefulPartitionedCall+conv2d_transpose_23/StatefulPartitionedCall2Z
+conv2d_transpose_24/StatefulPartitionedCall+conv2d_transpose_24/StatefulPartitionedCall2Z
+conv2d_transpose_25/StatefulPartitionedCall+conv2d_transpose_25/StatefulPartitionedCall2Z
+conv2d_transpose_26/StatefulPartitionedCall+conv2d_transpose_26/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2Z
+conv2d_transpose_30/StatefulPartitionedCall+conv2d_transpose_30/StatefulPartitionedCall2Z
+conv2d_transpose_31/StatefulPartitionedCall+conv2d_transpose_31/StatefulPartitionedCall2Z
+conv2d_transpose_32/StatefulPartitionedCall+conv2d_transpose_32/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
з 

O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_136162

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_59_layer_call_fn_140199

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_136654
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_139647

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
н
Ё
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_139419

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Х
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_139893

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
#

O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_139147

inputsC
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
: *
dtype0н
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_139305

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_136083

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
	
ж
7__inference_batch_normalization_56_layer_call_fn_139844

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_136299
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Х
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_139437

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_47_layer_call_and_return_conditional_losses_136736

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ *
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Щ
Љ
4__inference_conv2d_transpose_31_layer_call_fn_140140

inputs!
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_31_layer_call_and_return_conditional_losses_136594
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

С
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_140121

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
л 

O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_136378

inputsC
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Щ
Љ
4__inference_conv2d_transpose_26_layer_call_fn_139570

inputs!
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_136054
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

С
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_136006

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
з 

O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_136054

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Щ
K
/__inference_leaky_re_lu_47_layer_call_fn_139214

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_47_layer_call_and_return_conditional_losses_136736h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
з 

O__inference_conv2d_transpose_31_layer_call_and_return_conditional_losses_140173

inputsB
(conv2d_transpose_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

С
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_135682

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_140103

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_50_layer_call_fn_139173

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_135682
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_47_layer_call_and_return_conditional_losses_139219

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ *
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

С
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_136654

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Щ
K
/__inference_leaky_re_lu_51_layer_call_fn_139670

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_136820h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
п 

O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_139831

inputsC
(conv2d_transpose_readvariableop_resource: .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
B :y
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
: *
dtype0н
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
з 

O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_139603

inputsB
(conv2d_transpose_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_54_layer_call_fn_139616

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_136083
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
л 

O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_135946

inputsC
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
по
У<
C__inference_decoder_layer_call_and_return_conditional_losses_138737

inputsW
<conv2d_transpose_22_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_22_biasadd_readvariableop_resource: <
.batch_normalization_50_readvariableop_resource: >
0batch_normalization_50_readvariableop_1_resource: M
?batch_normalization_50_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_23_conv2d_transpose_readvariableop_resource:@ A
3conv2d_transpose_23_biasadd_readvariableop_resource:@<
.batch_normalization_51_readvariableop_resource:@>
0batch_normalization_51_readvariableop_1_resource:@M
?batch_normalization_51_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource:@W
<conv2d_transpose_24_conv2d_transpose_readvariableop_resource:@B
3conv2d_transpose_24_biasadd_readvariableop_resource:	=
.batch_normalization_52_readvariableop_resource:	?
0batch_normalization_52_readvariableop_1_resource:	N
?batch_normalization_52_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource:	W
<conv2d_transpose_25_conv2d_transpose_readvariableop_resource:@A
3conv2d_transpose_25_biasadd_readvariableop_resource:@<
.batch_normalization_53_readvariableop_resource:@>
0batch_normalization_53_readvariableop_1_resource:@M
?batch_normalization_53_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_26_conv2d_transpose_readvariableop_resource:@@A
3conv2d_transpose_26_biasadd_readvariableop_resource:@<
.batch_normalization_54_readvariableop_resource:@>
0batch_normalization_54_readvariableop_1_resource:@M
?batch_normalization_54_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_27_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_27_biasadd_readvariableop_resource: <
.batch_normalization_55_readvariableop_resource: >
0batch_normalization_55_readvariableop_1_resource: M
?batch_normalization_55_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource: W
<conv2d_transpose_28_conv2d_transpose_readvariableop_resource: B
3conv2d_transpose_28_biasadd_readvariableop_resource:	=
.batch_normalization_56_readvariableop_resource:	?
0batch_normalization_56_readvariableop_1_resource:	N
?batch_normalization_56_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource:	W
<conv2d_transpose_29_conv2d_transpose_readvariableop_resource:@A
3conv2d_transpose_29_biasadd_readvariableop_resource:@<
.batch_normalization_57_readvariableop_resource:@>
0batch_normalization_57_readvariableop_1_resource:@M
?batch_normalization_57_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_30_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_30_biasadd_readvariableop_resource: <
.batch_normalization_58_readvariableop_resource: >
0batch_normalization_58_readvariableop_1_resource: M
?batch_normalization_58_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_31_conv2d_transpose_readvariableop_resource:  A
3conv2d_transpose_31_biasadd_readvariableop_resource: <
.batch_normalization_59_readvariableop_resource: >
0batch_normalization_59_readvariableop_1_resource: M
?batch_normalization_59_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_59_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_32_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_32_biasadd_readvariableop_resource:
identityЂ6batch_normalization_50/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_50/ReadVariableOpЂ'batch_normalization_50/ReadVariableOp_1Ђ6batch_normalization_51/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_51/ReadVariableOpЂ'batch_normalization_51/ReadVariableOp_1Ђ6batch_normalization_52/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_52/ReadVariableOpЂ'batch_normalization_52/ReadVariableOp_1Ђ6batch_normalization_53/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_53/ReadVariableOpЂ'batch_normalization_53/ReadVariableOp_1Ђ6batch_normalization_54/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_54/ReadVariableOpЂ'batch_normalization_54/ReadVariableOp_1Ђ6batch_normalization_55/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_55/ReadVariableOpЂ'batch_normalization_55/ReadVariableOp_1Ђ6batch_normalization_56/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_56/ReadVariableOpЂ'batch_normalization_56/ReadVariableOp_1Ђ6batch_normalization_57/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_57/ReadVariableOpЂ'batch_normalization_57/ReadVariableOp_1Ђ6batch_normalization_58/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_58/ReadVariableOpЂ'batch_normalization_58/ReadVariableOp_1Ђ6batch_normalization_59/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_59/ReadVariableOpЂ'batch_normalization_59/ReadVariableOp_1Ђ*conv2d_transpose_22/BiasAdd/ReadVariableOpЂ3conv2d_transpose_22/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_23/BiasAdd/ReadVariableOpЂ3conv2d_transpose_23/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_24/BiasAdd/ReadVariableOpЂ3conv2d_transpose_24/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_25/BiasAdd/ReadVariableOpЂ3conv2d_transpose_25/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_26/BiasAdd/ReadVariableOpЂ3conv2d_transpose_26/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_27/BiasAdd/ReadVariableOpЂ3conv2d_transpose_27/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_28/BiasAdd/ReadVariableOpЂ3conv2d_transpose_28/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_29/BiasAdd/ReadVariableOpЂ3conv2d_transpose_29/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_30/BiasAdd/ReadVariableOpЂ3conv2d_transpose_30/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_31/BiasAdd/ReadVariableOpЂ3conv2d_transpose_31/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_32/BiasAdd/ReadVariableOpЂ3conv2d_transpose_32/conv2d_transpose/ReadVariableOpO
conv2d_transpose_22/ShapeShapeinputs*
T0*
_output_shapes
:q
'conv2d_transpose_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_22/strided_sliceStridedSlice"conv2d_transpose_22/Shape:output:00conv2d_transpose_22/strided_slice/stack:output:02conv2d_transpose_22/strided_slice/stack_1:output:02conv2d_transpose_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_22/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_22/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_22/stack/3Const*
_output_shapes
: *
dtype0*
value	B : э
conv2d_transpose_22/stackPack*conv2d_transpose_22/strided_slice:output:0$conv2d_transpose_22/stack/1:output:0$conv2d_transpose_22/stack/2:output:0$conv2d_transpose_22/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_22/strided_slice_1StridedSlice"conv2d_transpose_22/stack:output:02conv2d_transpose_22/strided_slice_1/stack:output:04conv2d_transpose_22/strided_slice_1/stack_1:output:04conv2d_transpose_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_22/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_22_conv2d_transpose_readvariableop_resource*'
_output_shapes
: *
dtype0
$conv2d_transpose_22/conv2d_transposeConv2DBackpropInput"conv2d_transpose_22/stack:output:0;conv2d_transpose_22/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides

*conv2d_transpose_22/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0У
conv2d_transpose_22/BiasAddBiasAdd-conv2d_transpose_22/conv2d_transpose:output:02conv2d_transpose_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 
%batch_normalization_50/ReadVariableOpReadVariableOp.batch_normalization_50_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_50/ReadVariableOp_1ReadVariableOp0batch_normalization_50_readvariableop_1_resource*
_output_shapes
: *
dtype0В
6batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ж
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ч
'batch_normalization_50/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_22/BiasAdd:output:0-batch_normalization_50/ReadVariableOp:value:0/batch_normalization_50/ReadVariableOp_1:value:0>batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 
leaky_re_lu_47/LeakyRelu	LeakyRelu+batch_normalization_50/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ *
alpha%>o
conv2d_transpose_23/ShapeShape&leaky_re_lu_47/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_23/strided_sliceStridedSlice"conv2d_transpose_23/Shape:output:00conv2d_transpose_23/strided_slice/stack:output:02conv2d_transpose_23/strided_slice/stack_1:output:02conv2d_transpose_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_23/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_23/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_23/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@э
conv2d_transpose_23/stackPack*conv2d_transpose_23/strided_slice:output:0$conv2d_transpose_23/stack/1:output:0$conv2d_transpose_23/stack/2:output:0$conv2d_transpose_23/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_23/strided_slice_1StridedSlice"conv2d_transpose_23/stack:output:02conv2d_transpose_23/strided_slice_1/stack:output:04conv2d_transpose_23/strided_slice_1/stack_1:output:04conv2d_transpose_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_23/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0І
$conv2d_transpose_23/conv2d_transposeConv2DBackpropInput"conv2d_transpose_23/stack:output:0;conv2d_transpose_23/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_47/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

*conv2d_transpose_23/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0У
conv2d_transpose_23/BiasAddBiasAdd-conv2d_transpose_23/conv2d_transpose:output:02conv2d_transpose_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
%batch_normalization_51/ReadVariableOpReadVariableOp.batch_normalization_51_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_51/ReadVariableOp_1ReadVariableOp0batch_normalization_51_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
6batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ч
'batch_normalization_51/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_23/BiasAdd:output:0-batch_normalization_51/ReadVariableOp:value:0/batch_normalization_51/ReadVariableOp_1:value:0>batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 
leaky_re_lu_48/LeakyRelu	LeakyRelu+batch_normalization_51/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@*
alpha%>o
conv2d_transpose_24/ShapeShape&leaky_re_lu_48/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_24/strided_sliceStridedSlice"conv2d_transpose_24/Shape:output:00conv2d_transpose_24/strided_slice/stack:output:02conv2d_transpose_24/strided_slice/stack_1:output:02conv2d_transpose_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_24/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_24/stack/2Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_24/stack/3Const*
_output_shapes
: *
dtype0*
value
B :э
conv2d_transpose_24/stackPack*conv2d_transpose_24/strided_slice:output:0$conv2d_transpose_24/stack/1:output:0$conv2d_transpose_24/stack/2:output:0$conv2d_transpose_24/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_24/strided_slice_1StridedSlice"conv2d_transpose_24/stack:output:02conv2d_transpose_24/strided_slice_1/stack:output:04conv2d_transpose_24/strided_slice_1/stack_1:output:04conv2d_transpose_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_24/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_24_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0Ї
$conv2d_transpose_24/conv2d_transposeConv2DBackpropInput"conv2d_transpose_24/stack:output:0;conv2d_transpose_24/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_48/LeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

*conv2d_transpose_24/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ф
conv2d_transpose_24/BiasAddBiasAdd-conv2d_transpose_24/conv2d_transpose:output:02conv2d_transpose_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
%batch_normalization_52/ReadVariableOpReadVariableOp.batch_normalization_52_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_52/ReadVariableOp_1ReadVariableOp0batch_normalization_52_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ь
'batch_normalization_52/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_24/BiasAdd:output:0-batch_normalization_52/ReadVariableOp:value:0/batch_normalization_52/ReadVariableOp_1:value:0>batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 
leaky_re_lu_49/LeakyRelu	LeakyRelu+batch_normalization_52/FusedBatchNormV3:y:0*0
_output_shapes
:џџџџџџџџџ*
alpha%>o
conv2d_transpose_25/ShapeShape&leaky_re_lu_49/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_25/strided_sliceStridedSlice"conv2d_transpose_25/Shape:output:00conv2d_transpose_25/strided_slice/stack:output:02conv2d_transpose_25/strided_slice/stack_1:output:02conv2d_transpose_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_25/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_25/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_25/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@э
conv2d_transpose_25/stackPack*conv2d_transpose_25/strided_slice:output:0$conv2d_transpose_25/stack/1:output:0$conv2d_transpose_25/stack/2:output:0$conv2d_transpose_25/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_25/strided_slice_1StridedSlice"conv2d_transpose_25/stack:output:02conv2d_transpose_25/strided_slice_1/stack:output:04conv2d_transpose_25/strided_slice_1/stack_1:output:04conv2d_transpose_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_25/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_25_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0І
$conv2d_transpose_25/conv2d_transposeConv2DBackpropInput"conv2d_transpose_25/stack:output:0;conv2d_transpose_25/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_49/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

*conv2d_transpose_25/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0У
conv2d_transpose_25/BiasAddBiasAdd-conv2d_transpose_25/conv2d_transpose:output:02conv2d_transpose_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
%batch_normalization_53/ReadVariableOpReadVariableOp.batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_53/ReadVariableOp_1ReadVariableOp0batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
6batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ч
'batch_normalization_53/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_25/BiasAdd:output:0-batch_normalization_53/ReadVariableOp:value:0/batch_normalization_53/ReadVariableOp_1:value:0>batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 
leaky_re_lu_50/LeakyRelu	LeakyRelu+batch_normalization_53/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@*
alpha%>o
conv2d_transpose_26/ShapeShape&leaky_re_lu_50/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_26/strided_sliceStridedSlice"conv2d_transpose_26/Shape:output:00conv2d_transpose_26/strided_slice/stack:output:02conv2d_transpose_26/strided_slice/stack_1:output:02conv2d_transpose_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_26/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_26/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_26/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@э
conv2d_transpose_26/stackPack*conv2d_transpose_26/strided_slice:output:0$conv2d_transpose_26/stack/1:output:0$conv2d_transpose_26/stack/2:output:0$conv2d_transpose_26/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_26/strided_slice_1StridedSlice"conv2d_transpose_26/stack:output:02conv2d_transpose_26/strided_slice_1/stack:output:04conv2d_transpose_26/strided_slice_1/stack_1:output:04conv2d_transpose_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_26/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_26_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0І
$conv2d_transpose_26/conv2d_transposeConv2DBackpropInput"conv2d_transpose_26/stack:output:0;conv2d_transpose_26/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_50/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

*conv2d_transpose_26/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0У
conv2d_transpose_26/BiasAddBiasAdd-conv2d_transpose_26/conv2d_transpose:output:02conv2d_transpose_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
%batch_normalization_54/ReadVariableOpReadVariableOp.batch_normalization_54_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_54/ReadVariableOp_1ReadVariableOp0batch_normalization_54_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
6batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ч
'batch_normalization_54/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_26/BiasAdd:output:0-batch_normalization_54/ReadVariableOp:value:0/batch_normalization_54/ReadVariableOp_1:value:0>batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 
leaky_re_lu_51/LeakyRelu	LeakyRelu+batch_normalization_54/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@*
alpha%>o
conv2d_transpose_27/ShapeShape&leaky_re_lu_51/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_27/strided_sliceStridedSlice"conv2d_transpose_27/Shape:output:00conv2d_transpose_27/strided_slice/stack:output:02conv2d_transpose_27/strided_slice/stack_1:output:02conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value	B : э
conv2d_transpose_27/stackPack*conv2d_transpose_27/strided_slice:output:0$conv2d_transpose_27/stack/1:output:0$conv2d_transpose_27/stack/2:output:0$conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_27/strided_slice_1StridedSlice"conv2d_transpose_27/stack:output:02conv2d_transpose_27/strided_slice_1/stack:output:04conv2d_transpose_27/strided_slice_1/stack_1:output:04conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0І
$conv2d_transpose_27/conv2d_transposeConv2DBackpropInput"conv2d_transpose_27/stack:output:0;conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_51/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

*conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0У
conv2d_transpose_27/BiasAddBiasAdd-conv2d_transpose_27/conv2d_transpose:output:02conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 
%batch_normalization_55/ReadVariableOpReadVariableOp.batch_normalization_55_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_55/ReadVariableOp_1ReadVariableOp0batch_normalization_55_readvariableop_1_resource*
_output_shapes
: *
dtype0В
6batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ж
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ч
'batch_normalization_55/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_27/BiasAdd:output:0-batch_normalization_55/ReadVariableOp:value:0/batch_normalization_55/ReadVariableOp_1:value:0>batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 
leaky_re_lu_52/LeakyRelu	LeakyRelu+batch_normalization_55/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ *
alpha%>o
conv2d_transpose_28/ShapeShape&leaky_re_lu_52/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_28/strided_sliceStridedSlice"conv2d_transpose_28/Shape:output:00conv2d_transpose_28/strided_slice/stack:output:02conv2d_transpose_28/strided_slice/stack_1:output:02conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ^
conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value
B :э
conv2d_transpose_28/stackPack*conv2d_transpose_28/strided_slice:output:0$conv2d_transpose_28/stack/1:output:0$conv2d_transpose_28/stack/2:output:0$conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_28/strided_slice_1StridedSlice"conv2d_transpose_28/stack:output:02conv2d_transpose_28/strided_slice_1/stack:output:04conv2d_transpose_28/strided_slice_1/stack_1:output:04conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*'
_output_shapes
: *
dtype0Ї
$conv2d_transpose_28/conv2d_transposeConv2DBackpropInput"conv2d_transpose_28/stack:output:0;conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_52/LeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides

*conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ф
conv2d_transpose_28/BiasAddBiasAdd-conv2d_transpose_28/conv2d_transpose:output:02conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  
%batch_normalization_56/ReadVariableOpReadVariableOp.batch_normalization_56_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_56/ReadVariableOp_1ReadVariableOp0batch_normalization_56_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_56/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_56_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_56_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ь
'batch_normalization_56/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_28/BiasAdd:output:0-batch_normalization_56/ReadVariableOp:value:0/batch_normalization_56/ReadVariableOp_1:value:0>batch_normalization_56/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ  :::::*
epsilon%o:*
is_training( 
leaky_re_lu_53/LeakyRelu	LeakyRelu+batch_normalization_56/FusedBatchNormV3:y:0*0
_output_shapes
:џџџџџџџџџ  *
alpha%>o
conv2d_transpose_29/ShapeShape&leaky_re_lu_53/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_29/strided_sliceStridedSlice"conv2d_transpose_29/Shape:output:00conv2d_transpose_29/strided_slice/stack:output:02conv2d_transpose_29/strided_slice/stack_1:output:02conv2d_transpose_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_29/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_29/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_29/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@э
conv2d_transpose_29/stackPack*conv2d_transpose_29/strided_slice:output:0$conv2d_transpose_29/stack/1:output:0$conv2d_transpose_29/stack/2:output:0$conv2d_transpose_29/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_29/strided_slice_1StridedSlice"conv2d_transpose_29/stack:output:02conv2d_transpose_29/strided_slice_1/stack:output:04conv2d_transpose_29/strided_slice_1/stack_1:output:04conv2d_transpose_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_29/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_29_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0І
$conv2d_transpose_29/conv2d_transposeConv2DBackpropInput"conv2d_transpose_29/stack:output:0;conv2d_transpose_29/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_53/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  @*
paddingSAME*
strides

*conv2d_transpose_29/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0У
conv2d_transpose_29/BiasAddBiasAdd-conv2d_transpose_29/conv2d_transpose:output:02conv2d_transpose_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  @
%batch_normalization_57/ReadVariableOpReadVariableOp.batch_normalization_57_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_57/ReadVariableOp_1ReadVariableOp0batch_normalization_57_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
6batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ч
'batch_normalization_57/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_29/BiasAdd:output:0-batch_normalization_57/ReadVariableOp:value:0/batch_normalization_57/ReadVariableOp_1:value:0>batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  @:@:@:@:@:*
epsilon%o:*
is_training( 
leaky_re_lu_54/LeakyRelu	LeakyRelu+batch_normalization_57/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ  @*
alpha%>o
conv2d_transpose_30/ShapeShape&leaky_re_lu_54/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_30/strided_sliceStridedSlice"conv2d_transpose_30/Shape:output:00conv2d_transpose_30/strided_slice/stack:output:02conv2d_transpose_30/strided_slice/stack_1:output:02conv2d_transpose_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_30/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_30/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@]
conv2d_transpose_30/stack/3Const*
_output_shapes
: *
dtype0*
value	B : э
conv2d_transpose_30/stackPack*conv2d_transpose_30/strided_slice:output:0$conv2d_transpose_30/stack/1:output:0$conv2d_transpose_30/stack/2:output:0$conv2d_transpose_30/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_30/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_30/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_30/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_30/strided_slice_1StridedSlice"conv2d_transpose_30/stack:output:02conv2d_transpose_30/strided_slice_1/stack:output:04conv2d_transpose_30/strided_slice_1/stack_1:output:04conv2d_transpose_30/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_30/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_30_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0І
$conv2d_transpose_30/conv2d_transposeConv2DBackpropInput"conv2d_transpose_30/stack:output:0;conv2d_transpose_30/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_54/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@ *
paddingSAME*
strides

*conv2d_transpose_30/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_30_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0У
conv2d_transpose_30/BiasAddBiasAdd-conv2d_transpose_30/conv2d_transpose:output:02conv2d_transpose_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@ 
%batch_normalization_58/ReadVariableOpReadVariableOp.batch_normalization_58_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_58/ReadVariableOp_1ReadVariableOp0batch_normalization_58_readvariableop_1_resource*
_output_shapes
: *
dtype0В
6batch_normalization_58/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_58_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ж
8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_58_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ч
'batch_normalization_58/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_30/BiasAdd:output:0-batch_normalization_58/ReadVariableOp:value:0/batch_normalization_58/ReadVariableOp_1:value:0>batch_normalization_58/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@ : : : : :*
epsilon%o:*
is_training( 
leaky_re_lu_55/LeakyRelu	LeakyRelu+batch_normalization_58/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@@ *
alpha%>o
conv2d_transpose_31/ShapeShape&leaky_re_lu_55/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_31/strided_sliceStridedSlice"conv2d_transpose_31/Shape:output:00conv2d_transpose_31/strided_slice/stack:output:02conv2d_transpose_31/strided_slice/stack_1:output:02conv2d_transpose_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_31/stack/1Const*
_output_shapes
: *
dtype0*
value
B :^
conv2d_transpose_31/stack/2Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_31/stack/3Const*
_output_shapes
: *
dtype0*
value	B : э
conv2d_transpose_31/stackPack*conv2d_transpose_31/strided_slice:output:0$conv2d_transpose_31/stack/1:output:0$conv2d_transpose_31/stack/2:output:0$conv2d_transpose_31/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_31/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_31/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_31/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_31/strided_slice_1StridedSlice"conv2d_transpose_31/stack:output:02conv2d_transpose_31/strided_slice_1/stack:output:04conv2d_transpose_31/strided_slice_1/stack_1:output:04conv2d_transpose_31/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_31/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_31_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0Ј
$conv2d_transpose_31/conv2d_transposeConv2DBackpropInput"conv2d_transpose_31/stack:output:0;conv2d_transpose_31/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_55/LeakyRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

*conv2d_transpose_31/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Х
conv2d_transpose_31/BiasAddBiasAdd-conv2d_transpose_31/conv2d_transpose:output:02conv2d_transpose_31/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ 
%batch_normalization_59/ReadVariableOpReadVariableOp.batch_normalization_59_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_59/ReadVariableOp_1ReadVariableOp0batch_normalization_59_readvariableop_1_resource*
_output_shapes
: *
dtype0В
6batch_normalization_59/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_59_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ж
8batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_59_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Щ
'batch_normalization_59/FusedBatchNormV3FusedBatchNormV3$conv2d_transpose_31/BiasAdd:output:0-batch_normalization_59/ReadVariableOp:value:0/batch_normalization_59/ReadVariableOp_1:value:0>batch_normalization_59/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 
leaky_re_lu_56/LeakyRelu	LeakyRelu+batch_normalization_59/FusedBatchNormV3:y:0*1
_output_shapes
:џџџџџџџџџ *
alpha%>o
conv2d_transpose_32/ShapeShape&leaky_re_lu_56/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_32/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_32/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_32/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_32/strided_sliceStridedSlice"conv2d_transpose_32/Shape:output:00conv2d_transpose_32/strided_slice/stack:output:02conv2d_transpose_32/strided_slice/stack_1:output:02conv2d_transpose_32/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_32/stack/1Const*
_output_shapes
: *
dtype0*
value
B :^
conv2d_transpose_32/stack/2Const*
_output_shapes
: *
dtype0*
value
B :]
conv2d_transpose_32/stack/3Const*
_output_shapes
: *
dtype0*
value	B :э
conv2d_transpose_32/stackPack*conv2d_transpose_32/strided_slice:output:0$conv2d_transpose_32/stack/1:output:0$conv2d_transpose_32/stack/2:output:0$conv2d_transpose_32/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_32/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_32/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_32/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_32/strided_slice_1StridedSlice"conv2d_transpose_32/stack:output:02conv2d_transpose_32/strided_slice_1/stack:output:04conv2d_transpose_32/strided_slice_1/stack_1:output:04conv2d_transpose_32/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_32/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_32_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ј
$conv2d_transpose_32/conv2d_transposeConv2DBackpropInput"conv2d_transpose_32/stack:output:0;conv2d_transpose_32/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_56/LeakyRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

*conv2d_transpose_32/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Х
conv2d_transpose_32/BiasAddBiasAdd-conv2d_transpose_32/conv2d_transpose:output:02conv2d_transpose_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ
conv2d_transpose_32/SigmoidSigmoid$conv2d_transpose_32/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџx
IdentityIdentityconv2d_transpose_32/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџУ
NoOpNoOp7^batch_normalization_50/FusedBatchNormV3/ReadVariableOp9^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_50/ReadVariableOp(^batch_normalization_50/ReadVariableOp_17^batch_normalization_51/FusedBatchNormV3/ReadVariableOp9^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_51/ReadVariableOp(^batch_normalization_51/ReadVariableOp_17^batch_normalization_52/FusedBatchNormV3/ReadVariableOp9^batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_52/ReadVariableOp(^batch_normalization_52/ReadVariableOp_17^batch_normalization_53/FusedBatchNormV3/ReadVariableOp9^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_53/ReadVariableOp(^batch_normalization_53/ReadVariableOp_17^batch_normalization_54/FusedBatchNormV3/ReadVariableOp9^batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_54/ReadVariableOp(^batch_normalization_54/ReadVariableOp_17^batch_normalization_55/FusedBatchNormV3/ReadVariableOp9^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_55/ReadVariableOp(^batch_normalization_55/ReadVariableOp_17^batch_normalization_56/FusedBatchNormV3/ReadVariableOp9^batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_56/ReadVariableOp(^batch_normalization_56/ReadVariableOp_17^batch_normalization_57/FusedBatchNormV3/ReadVariableOp9^batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_57/ReadVariableOp(^batch_normalization_57/ReadVariableOp_17^batch_normalization_58/FusedBatchNormV3/ReadVariableOp9^batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_58/ReadVariableOp(^batch_normalization_58/ReadVariableOp_17^batch_normalization_59/FusedBatchNormV3/ReadVariableOp9^batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_59/ReadVariableOp(^batch_normalization_59/ReadVariableOp_1+^conv2d_transpose_22/BiasAdd/ReadVariableOp4^conv2d_transpose_22/conv2d_transpose/ReadVariableOp+^conv2d_transpose_23/BiasAdd/ReadVariableOp4^conv2d_transpose_23/conv2d_transpose/ReadVariableOp+^conv2d_transpose_24/BiasAdd/ReadVariableOp4^conv2d_transpose_24/conv2d_transpose/ReadVariableOp+^conv2d_transpose_25/BiasAdd/ReadVariableOp4^conv2d_transpose_25/conv2d_transpose/ReadVariableOp+^conv2d_transpose_26/BiasAdd/ReadVariableOp4^conv2d_transpose_26/conv2d_transpose/ReadVariableOp+^conv2d_transpose_27/BiasAdd/ReadVariableOp4^conv2d_transpose_27/conv2d_transpose/ReadVariableOp+^conv2d_transpose_28/BiasAdd/ReadVariableOp4^conv2d_transpose_28/conv2d_transpose/ReadVariableOp+^conv2d_transpose_29/BiasAdd/ReadVariableOp4^conv2d_transpose_29/conv2d_transpose/ReadVariableOp+^conv2d_transpose_30/BiasAdd/ReadVariableOp4^conv2d_transpose_30/conv2d_transpose/ReadVariableOp+^conv2d_transpose_31/BiasAdd/ReadVariableOp4^conv2d_transpose_31/conv2d_transpose/ReadVariableOp+^conv2d_transpose_32/BiasAdd/ReadVariableOp4^conv2d_transpose_32/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp6batch_normalization_50/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_18batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_50/ReadVariableOp%batch_normalization_50/ReadVariableOp2R
'batch_normalization_50/ReadVariableOp_1'batch_normalization_50/ReadVariableOp_12p
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp6batch_normalization_51/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_18batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_51/ReadVariableOp%batch_normalization_51/ReadVariableOp2R
'batch_normalization_51/ReadVariableOp_1'batch_normalization_51/ReadVariableOp_12p
6batch_normalization_52/FusedBatchNormV3/ReadVariableOp6batch_normalization_52/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_18batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_52/ReadVariableOp%batch_normalization_52/ReadVariableOp2R
'batch_normalization_52/ReadVariableOp_1'batch_normalization_52/ReadVariableOp_12p
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp6batch_normalization_53/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_18batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_53/ReadVariableOp%batch_normalization_53/ReadVariableOp2R
'batch_normalization_53/ReadVariableOp_1'batch_normalization_53/ReadVariableOp_12p
6batch_normalization_54/FusedBatchNormV3/ReadVariableOp6batch_normalization_54/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_18batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_54/ReadVariableOp%batch_normalization_54/ReadVariableOp2R
'batch_normalization_54/ReadVariableOp_1'batch_normalization_54/ReadVariableOp_12p
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp6batch_normalization_55/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_18batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_55/ReadVariableOp%batch_normalization_55/ReadVariableOp2R
'batch_normalization_55/ReadVariableOp_1'batch_normalization_55/ReadVariableOp_12p
6batch_normalization_56/FusedBatchNormV3/ReadVariableOp6batch_normalization_56/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_56/FusedBatchNormV3/ReadVariableOp_18batch_normalization_56/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_56/ReadVariableOp%batch_normalization_56/ReadVariableOp2R
'batch_normalization_56/ReadVariableOp_1'batch_normalization_56/ReadVariableOp_12p
6batch_normalization_57/FusedBatchNormV3/ReadVariableOp6batch_normalization_57/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_57/FusedBatchNormV3/ReadVariableOp_18batch_normalization_57/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_57/ReadVariableOp%batch_normalization_57/ReadVariableOp2R
'batch_normalization_57/ReadVariableOp_1'batch_normalization_57/ReadVariableOp_12p
6batch_normalization_58/FusedBatchNormV3/ReadVariableOp6batch_normalization_58/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_58/FusedBatchNormV3/ReadVariableOp_18batch_normalization_58/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_58/ReadVariableOp%batch_normalization_58/ReadVariableOp2R
'batch_normalization_58/ReadVariableOp_1'batch_normalization_58/ReadVariableOp_12p
6batch_normalization_59/FusedBatchNormV3/ReadVariableOp6batch_normalization_59/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_59/FusedBatchNormV3/ReadVariableOp_18batch_normalization_59/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_59/ReadVariableOp%batch_normalization_59/ReadVariableOp2R
'batch_normalization_59/ReadVariableOp_1'batch_normalization_59/ReadVariableOp_12X
*conv2d_transpose_22/BiasAdd/ReadVariableOp*conv2d_transpose_22/BiasAdd/ReadVariableOp2j
3conv2d_transpose_22/conv2d_transpose/ReadVariableOp3conv2d_transpose_22/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_23/BiasAdd/ReadVariableOp*conv2d_transpose_23/BiasAdd/ReadVariableOp2j
3conv2d_transpose_23/conv2d_transpose/ReadVariableOp3conv2d_transpose_23/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_24/BiasAdd/ReadVariableOp*conv2d_transpose_24/BiasAdd/ReadVariableOp2j
3conv2d_transpose_24/conv2d_transpose/ReadVariableOp3conv2d_transpose_24/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_25/BiasAdd/ReadVariableOp*conv2d_transpose_25/BiasAdd/ReadVariableOp2j
3conv2d_transpose_25/conv2d_transpose/ReadVariableOp3conv2d_transpose_25/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_26/BiasAdd/ReadVariableOp*conv2d_transpose_26/BiasAdd/ReadVariableOp2j
3conv2d_transpose_26/conv2d_transpose/ReadVariableOp3conv2d_transpose_26/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_27/BiasAdd/ReadVariableOp*conv2d_transpose_27/BiasAdd/ReadVariableOp2j
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp3conv2d_transpose_27/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_28/BiasAdd/ReadVariableOp*conv2d_transpose_28/BiasAdd/ReadVariableOp2j
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp3conv2d_transpose_28/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_29/BiasAdd/ReadVariableOp*conv2d_transpose_29/BiasAdd/ReadVariableOp2j
3conv2d_transpose_29/conv2d_transpose/ReadVariableOp3conv2d_transpose_29/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_30/BiasAdd/ReadVariableOp*conv2d_transpose_30/BiasAdd/ReadVariableOp2j
3conv2d_transpose_30/conv2d_transpose/ReadVariableOp3conv2d_transpose_30/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_31/BiasAdd/ReadVariableOp*conv2d_transpose_31/BiasAdd/ReadVariableOp2j
3conv2d_transpose_31/conv2d_transpose/ReadVariableOp3conv2d_transpose_31/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_32/BiasAdd/ReadVariableOp*conv2d_transpose_32/BiasAdd/ReadVariableOp2j
3conv2d_transpose_32/conv2d_transpose/ReadVariableOp3conv2d_transpose_32/conv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_139191

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

С
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_139209

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_57_layer_call_fn_139971

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_136438
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

С
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_136438

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
э~

__inference__traced_save_140497
file_prefix9
5savev2_conv2d_transpose_22_kernel_read_readvariableop7
3savev2_conv2d_transpose_22_bias_read_readvariableop;
7savev2_batch_normalization_50_gamma_read_readvariableop:
6savev2_batch_normalization_50_beta_read_readvariableopA
=savev2_batch_normalization_50_moving_mean_read_readvariableopE
Asavev2_batch_normalization_50_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_23_kernel_read_readvariableop7
3savev2_conv2d_transpose_23_bias_read_readvariableop;
7savev2_batch_normalization_51_gamma_read_readvariableop:
6savev2_batch_normalization_51_beta_read_readvariableopA
=savev2_batch_normalization_51_moving_mean_read_readvariableopE
Asavev2_batch_normalization_51_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_24_kernel_read_readvariableop7
3savev2_conv2d_transpose_24_bias_read_readvariableop;
7savev2_batch_normalization_52_gamma_read_readvariableop:
6savev2_batch_normalization_52_beta_read_readvariableopA
=savev2_batch_normalization_52_moving_mean_read_readvariableopE
Asavev2_batch_normalization_52_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_25_kernel_read_readvariableop7
3savev2_conv2d_transpose_25_bias_read_readvariableop;
7savev2_batch_normalization_53_gamma_read_readvariableop:
6savev2_batch_normalization_53_beta_read_readvariableopA
=savev2_batch_normalization_53_moving_mean_read_readvariableopE
Asavev2_batch_normalization_53_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_26_kernel_read_readvariableop7
3savev2_conv2d_transpose_26_bias_read_readvariableop;
7savev2_batch_normalization_54_gamma_read_readvariableop:
6savev2_batch_normalization_54_beta_read_readvariableopA
=savev2_batch_normalization_54_moving_mean_read_readvariableopE
Asavev2_batch_normalization_54_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_27_kernel_read_readvariableop7
3savev2_conv2d_transpose_27_bias_read_readvariableop;
7savev2_batch_normalization_55_gamma_read_readvariableop:
6savev2_batch_normalization_55_beta_read_readvariableopA
=savev2_batch_normalization_55_moving_mean_read_readvariableopE
Asavev2_batch_normalization_55_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_28_kernel_read_readvariableop7
3savev2_conv2d_transpose_28_bias_read_readvariableop;
7savev2_batch_normalization_56_gamma_read_readvariableop:
6savev2_batch_normalization_56_beta_read_readvariableopA
=savev2_batch_normalization_56_moving_mean_read_readvariableopE
Asavev2_batch_normalization_56_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_29_kernel_read_readvariableop7
3savev2_conv2d_transpose_29_bias_read_readvariableop;
7savev2_batch_normalization_57_gamma_read_readvariableop:
6savev2_batch_normalization_57_beta_read_readvariableopA
=savev2_batch_normalization_57_moving_mean_read_readvariableopE
Asavev2_batch_normalization_57_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_30_kernel_read_readvariableop7
3savev2_conv2d_transpose_30_bias_read_readvariableop;
7savev2_batch_normalization_58_gamma_read_readvariableop:
6savev2_batch_normalization_58_beta_read_readvariableopA
=savev2_batch_normalization_58_moving_mean_read_readvariableopE
Asavev2_batch_normalization_58_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_31_kernel_read_readvariableop7
3savev2_conv2d_transpose_31_bias_read_readvariableop;
7savev2_batch_normalization_59_gamma_read_readvariableop:
6savev2_batch_normalization_59_beta_read_readvariableopA
=savev2_batch_normalization_59_moving_mean_read_readvariableopE
Asavev2_batch_normalization_59_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_32_kernel_read_readvariableop7
3savev2_conv2d_transpose_32_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*Л
valueБBЎ?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHю
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*
valueB?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_conv2d_transpose_22_kernel_read_readvariableop3savev2_conv2d_transpose_22_bias_read_readvariableop7savev2_batch_normalization_50_gamma_read_readvariableop6savev2_batch_normalization_50_beta_read_readvariableop=savev2_batch_normalization_50_moving_mean_read_readvariableopAsavev2_batch_normalization_50_moving_variance_read_readvariableop5savev2_conv2d_transpose_23_kernel_read_readvariableop3savev2_conv2d_transpose_23_bias_read_readvariableop7savev2_batch_normalization_51_gamma_read_readvariableop6savev2_batch_normalization_51_beta_read_readvariableop=savev2_batch_normalization_51_moving_mean_read_readvariableopAsavev2_batch_normalization_51_moving_variance_read_readvariableop5savev2_conv2d_transpose_24_kernel_read_readvariableop3savev2_conv2d_transpose_24_bias_read_readvariableop7savev2_batch_normalization_52_gamma_read_readvariableop6savev2_batch_normalization_52_beta_read_readvariableop=savev2_batch_normalization_52_moving_mean_read_readvariableopAsavev2_batch_normalization_52_moving_variance_read_readvariableop5savev2_conv2d_transpose_25_kernel_read_readvariableop3savev2_conv2d_transpose_25_bias_read_readvariableop7savev2_batch_normalization_53_gamma_read_readvariableop6savev2_batch_normalization_53_beta_read_readvariableop=savev2_batch_normalization_53_moving_mean_read_readvariableopAsavev2_batch_normalization_53_moving_variance_read_readvariableop5savev2_conv2d_transpose_26_kernel_read_readvariableop3savev2_conv2d_transpose_26_bias_read_readvariableop7savev2_batch_normalization_54_gamma_read_readvariableop6savev2_batch_normalization_54_beta_read_readvariableop=savev2_batch_normalization_54_moving_mean_read_readvariableopAsavev2_batch_normalization_54_moving_variance_read_readvariableop5savev2_conv2d_transpose_27_kernel_read_readvariableop3savev2_conv2d_transpose_27_bias_read_readvariableop7savev2_batch_normalization_55_gamma_read_readvariableop6savev2_batch_normalization_55_beta_read_readvariableop=savev2_batch_normalization_55_moving_mean_read_readvariableopAsavev2_batch_normalization_55_moving_variance_read_readvariableop5savev2_conv2d_transpose_28_kernel_read_readvariableop3savev2_conv2d_transpose_28_bias_read_readvariableop7savev2_batch_normalization_56_gamma_read_readvariableop6savev2_batch_normalization_56_beta_read_readvariableop=savev2_batch_normalization_56_moving_mean_read_readvariableopAsavev2_batch_normalization_56_moving_variance_read_readvariableop5savev2_conv2d_transpose_29_kernel_read_readvariableop3savev2_conv2d_transpose_29_bias_read_readvariableop7savev2_batch_normalization_57_gamma_read_readvariableop6savev2_batch_normalization_57_beta_read_readvariableop=savev2_batch_normalization_57_moving_mean_read_readvariableopAsavev2_batch_normalization_57_moving_variance_read_readvariableop5savev2_conv2d_transpose_30_kernel_read_readvariableop3savev2_conv2d_transpose_30_bias_read_readvariableop7savev2_batch_normalization_58_gamma_read_readvariableop6savev2_batch_normalization_58_beta_read_readvariableop=savev2_batch_normalization_58_moving_mean_read_readvariableopAsavev2_batch_normalization_58_moving_variance_read_readvariableop5savev2_conv2d_transpose_31_kernel_read_readvariableop3savev2_conv2d_transpose_31_bias_read_readvariableop7savev2_batch_normalization_59_gamma_read_readvariableop6savev2_batch_normalization_59_beta_read_readvariableop=savev2_batch_normalization_59_moving_mean_read_readvariableopAsavev2_batch_normalization_59_moving_variance_read_readvariableop5savev2_conv2d_transpose_32_kernel_read_readvariableop3savev2_conv2d_transpose_32_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *M
dtypesC
A2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0* 
_input_shapes
: : : : : : : :@ :@:@:@:@:@:@::::::@:@:@:@:@:@:@@:@:@:@:@:@: @: : : : : : ::::::@:@:@:@:@:@: @: : : : : :  : : : : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
: : 
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
:@ : 
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
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::-)
'
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
: @:  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: :-%)
'
_output_shapes
: :!&

_output_shapes	
::!'

_output_shapes	
::!(

_output_shapes	
::!)

_output_shapes	
::!*

_output_shapes	
::-+)
'
_output_shapes
:@: ,

_output_shapes
:@: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
:@:,1(
&
_output_shapes
: @: 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: :,7(
&
_output_shapes
:  : 8

_output_shapes
: : 9

_output_shapes
: : :

_output_shapes
: : ;

_output_shapes
: : <

_output_shapes
: :,=(
&
_output_shapes
: : >

_output_shapes
::?

_output_shapes
: 
	
ж
7__inference_batch_normalization_52_layer_call_fn_139388

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_135867
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Э
Ћ
4__inference_conv2d_transpose_28_layer_call_fn_139798

inputs"
unknown: 
	unknown_0:	
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_136270
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_59_layer_call_fn_140186

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_136623
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_139789

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ *
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_136820

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ@*
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_48_layer_call_and_return_conditional_losses_139333

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ@*
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
п 

O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_136270

inputsC
(conv2d_transpose_readvariableop_resource: .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
B :y
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
: *
dtype0н
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_139533

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Ф!

O__inference_conv2d_transpose_32_layer_call_and_return_conditional_losses_140288

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџp
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџt
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Ь
Њ
4__inference_conv2d_transpose_25_layer_call_fn_139456

inputs"
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_135946
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

С
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_135790

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Њ
Г
(__inference_decoder_layer_call_fn_138373

inputs"
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:@ 
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29: @

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: %

unknown_35: 

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	

unknown_40:	%

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47: @

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: $

unknown_53:  

unknown_54: 

unknown_55: 

unknown_56: 

unknown_57: 

unknown_58: $

unknown_59: 

unknown_60:
identityЂStatefulPartitionedCall	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ*L
_read_only_resource_inputs.
,*	
 !"%&'(+,-.1234789:=>*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_137410y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

А
$__inference_signature_wrapper_138115
input_6"
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:@ 
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29: @

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: %

unknown_35: 

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	

unknown_40:	%

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47: @

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: $

unknown_53:  

unknown_54: 

unknown_55: 

unknown_56: 

unknown_57: 

unknown_58: $

unknown_59: 

unknown_60:
identityЂStatefulPartitionedCall	
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_135581y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
з 

O__inference_conv2d_transpose_30_layer_call_and_return_conditional_losses_140059

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

С
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_139323

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Ь
Њ
4__inference_conv2d_transpose_22_layer_call_fn_139110

inputs"
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_135622
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Щ
K
/__inference_leaky_re_lu_54_layer_call_fn_140012

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_136883h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ  @:W S
/
_output_shapes
:џџџџџџџџџ  @
 
_user_specified_nameinputs
	
ж
7__inference_batch_normalization_56_layer_call_fn_139857

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_136330
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Щ
K
/__inference_leaky_re_lu_55_layer_call_fn_140126

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_136904h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@ :W S
/
_output_shapes
:џџџџџџџџџ@@ 
 
_user_specified_nameinputs

С
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_139551

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_139761

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

С
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_136114

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
б
K
/__inference_leaky_re_lu_56_layer_call_fn_140240

inputs
identityП
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_136925j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ :Y U
1
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_53_layer_call_fn_139502

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_135975
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_135975

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_140131

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ@@ *
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@ :W S
/
_output_shapes
:џџџџџџџџџ@@ 
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_140217

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_53_layer_call_fn_139515

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_136006
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_136925

inputs
identitya
	LeakyRelu	LeakyReluinputs*1
_output_shapes
:џџџџџџџџџ *
alpha%>i
IdentityIdentityLeakyRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ :Y U
1
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
з 

O__inference_conv2d_transpose_31_layer_call_and_return_conditional_losses_136594

inputsB
(conv2d_transpose_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

С
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_140235

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_135759

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_139675

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ@*
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_136407

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_51_layer_call_fn_139274

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_135759
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
#

O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_135622

inputsC
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
: *
dtype0н
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Э
Ћ
4__inference_conv2d_transpose_24_layer_call_fn_139342

inputs"
unknown:@
	unknown_0:	
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_135838
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_136191

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Щ
Љ
4__inference_conv2d_transpose_30_layer_call_fn_140026

inputs!
unknown: @
	unknown_0: 
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_30_layer_call_and_return_conditional_losses_136486
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
	
ж
7__inference_batch_normalization_52_layer_call_fn_139401

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_135898
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_136862

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:џџџџџџџџџ  *
alpha%>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ  :X T
0
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_57_layer_call_fn_139958

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_136407
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_55_layer_call_fn_139730

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_136191
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

С
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_139665

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Щ
K
/__inference_leaky_re_lu_50_layer_call_fn_139556

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_136799h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
з 

O__inference_conv2d_transpose_30_layer_call_and_return_conditional_losses_136486

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_139989

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
п 

O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_135838

inputsC
(conv2d_transpose_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :y
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0н
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџz
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Э
K
/__inference_leaky_re_lu_49_layer_call_fn_139442

inputs
identityО
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_136778i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Х
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_136330

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Э

R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_136623

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
н
Ё
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_139875

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
л 

O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_139489

inputsC
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

С
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_136546

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_136799

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ@*
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

С
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_139779

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
з 

O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_135730

inputsB
(conv2d_transpose_readvariableop_resource:@ -
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
С
Д
(__inference_decoder_layer_call_fn_137060
input_6"
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:@ 
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29: @

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: %

unknown_35: 

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	

unknown_40:	%

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47: @

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: $

unknown_53:  

unknown_54: 

unknown_55: 

unknown_56: 

unknown_57: 

unknown_58: $

unknown_59: 

unknown_60:
identityЂStatefulPartitionedCallЂ	
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_136933y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
ЛЊ
§
C__inference_decoder_layer_call_and_return_conditional_losses_136933

inputs5
conv2d_transpose_22_136717: (
conv2d_transpose_22_136719: +
batch_normalization_50_136722: +
batch_normalization_50_136724: +
batch_normalization_50_136726: +
batch_normalization_50_136728: 4
conv2d_transpose_23_136738:@ (
conv2d_transpose_23_136740:@+
batch_normalization_51_136743:@+
batch_normalization_51_136745:@+
batch_normalization_51_136747:@+
batch_normalization_51_136749:@5
conv2d_transpose_24_136759:@)
conv2d_transpose_24_136761:	,
batch_normalization_52_136764:	,
batch_normalization_52_136766:	,
batch_normalization_52_136768:	,
batch_normalization_52_136770:	5
conv2d_transpose_25_136780:@(
conv2d_transpose_25_136782:@+
batch_normalization_53_136785:@+
batch_normalization_53_136787:@+
batch_normalization_53_136789:@+
batch_normalization_53_136791:@4
conv2d_transpose_26_136801:@@(
conv2d_transpose_26_136803:@+
batch_normalization_54_136806:@+
batch_normalization_54_136808:@+
batch_normalization_54_136810:@+
batch_normalization_54_136812:@4
conv2d_transpose_27_136822: @(
conv2d_transpose_27_136824: +
batch_normalization_55_136827: +
batch_normalization_55_136829: +
batch_normalization_55_136831: +
batch_normalization_55_136833: 5
conv2d_transpose_28_136843: )
conv2d_transpose_28_136845:	,
batch_normalization_56_136848:	,
batch_normalization_56_136850:	,
batch_normalization_56_136852:	,
batch_normalization_56_136854:	5
conv2d_transpose_29_136864:@(
conv2d_transpose_29_136866:@+
batch_normalization_57_136869:@+
batch_normalization_57_136871:@+
batch_normalization_57_136873:@+
batch_normalization_57_136875:@4
conv2d_transpose_30_136885: @(
conv2d_transpose_30_136887: +
batch_normalization_58_136890: +
batch_normalization_58_136892: +
batch_normalization_58_136894: +
batch_normalization_58_136896: 4
conv2d_transpose_31_136906:  (
conv2d_transpose_31_136908: +
batch_normalization_59_136911: +
batch_normalization_59_136913: +
batch_normalization_59_136915: +
batch_normalization_59_136917: 4
conv2d_transpose_32_136927: (
conv2d_transpose_32_136929:
identityЂ.batch_normalization_50/StatefulPartitionedCallЂ.batch_normalization_51/StatefulPartitionedCallЂ.batch_normalization_52/StatefulPartitionedCallЂ.batch_normalization_53/StatefulPartitionedCallЂ.batch_normalization_54/StatefulPartitionedCallЂ.batch_normalization_55/StatefulPartitionedCallЂ.batch_normalization_56/StatefulPartitionedCallЂ.batch_normalization_57/StatefulPartitionedCallЂ.batch_normalization_58/StatefulPartitionedCallЂ.batch_normalization_59/StatefulPartitionedCallЂ+conv2d_transpose_22/StatefulPartitionedCallЂ+conv2d_transpose_23/StatefulPartitionedCallЂ+conv2d_transpose_24/StatefulPartitionedCallЂ+conv2d_transpose_25/StatefulPartitionedCallЂ+conv2d_transpose_26/StatefulPartitionedCallЂ+conv2d_transpose_27/StatefulPartitionedCallЂ+conv2d_transpose_28/StatefulPartitionedCallЂ+conv2d_transpose_29/StatefulPartitionedCallЂ+conv2d_transpose_30/StatefulPartitionedCallЂ+conv2d_transpose_31/StatefulPartitionedCallЂ+conv2d_transpose_32/StatefulPartitionedCallЄ
+conv2d_transpose_22/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_22_136717conv2d_transpose_22_136719*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_135622 
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_22/StatefulPartitionedCall:output:0batch_normalization_50_136722batch_normalization_50_136724batch_normalization_50_136726batch_normalization_50_136728*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_135651§
leaky_re_lu_47/PartitionedCallPartitionedCall7batch_normalization_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_47_layer_call_and_return_conditional_losses_136736Х
+conv2d_transpose_23/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_47/PartitionedCall:output:0conv2d_transpose_23_136738conv2d_transpose_23_136740*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_135730 
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_23/StatefulPartitionedCall:output:0batch_normalization_51_136743batch_normalization_51_136745batch_normalization_51_136747batch_normalization_51_136749*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_135759§
leaky_re_lu_48/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_48_layer_call_and_return_conditional_losses_136757Ц
+conv2d_transpose_24/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_48/PartitionedCall:output:0conv2d_transpose_24_136759conv2d_transpose_24_136761*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_135838Ё
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_24/StatefulPartitionedCall:output:0batch_normalization_52_136764batch_normalization_52_136766batch_normalization_52_136768batch_normalization_52_136770*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_135867ў
leaky_re_lu_49/PartitionedCallPartitionedCall7batch_normalization_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_136778Х
+conv2d_transpose_25/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_49/PartitionedCall:output:0conv2d_transpose_25_136780conv2d_transpose_25_136782*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_135946 
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_25/StatefulPartitionedCall:output:0batch_normalization_53_136785batch_normalization_53_136787batch_normalization_53_136789batch_normalization_53_136791*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_135975§
leaky_re_lu_50/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_136799Х
+conv2d_transpose_26/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_50/PartitionedCall:output:0conv2d_transpose_26_136801conv2d_transpose_26_136803*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_136054 
.batch_normalization_54/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_26/StatefulPartitionedCall:output:0batch_normalization_54_136806batch_normalization_54_136808batch_normalization_54_136810batch_normalization_54_136812*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_136083§
leaky_re_lu_51/PartitionedCallPartitionedCall7batch_normalization_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_136820Х
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_51/PartitionedCall:output:0conv2d_transpose_27_136822conv2d_transpose_27_136824*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_136162 
.batch_normalization_55/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0batch_normalization_55_136827batch_normalization_55_136829batch_normalization_55_136831batch_normalization_55_136833*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_136191§
leaky_re_lu_52/PartitionedCallPartitionedCall7batch_normalization_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_136841Ц
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_52/PartitionedCall:output:0conv2d_transpose_28_136843conv2d_transpose_28_136845*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_136270Ё
.batch_normalization_56/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0batch_normalization_56_136848batch_normalization_56_136850batch_normalization_56_136852batch_normalization_56_136854*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_136299ў
leaky_re_lu_53/PartitionedCallPartitionedCall7batch_normalization_56/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_136862Х
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_53/PartitionedCall:output:0conv2d_transpose_29_136864conv2d_transpose_29_136866*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_136378 
.batch_normalization_57/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_29/StatefulPartitionedCall:output:0batch_normalization_57_136869batch_normalization_57_136871batch_normalization_57_136873batch_normalization_57_136875*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_136407§
leaky_re_lu_54/PartitionedCallPartitionedCall7batch_normalization_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_136883Х
+conv2d_transpose_30/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_54/PartitionedCall:output:0conv2d_transpose_30_136885conv2d_transpose_30_136887*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_30_layer_call_and_return_conditional_losses_136486 
.batch_normalization_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_30/StatefulPartitionedCall:output:0batch_normalization_58_136890batch_normalization_58_136892batch_normalization_58_136894batch_normalization_58_136896*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_136515§
leaky_re_lu_55/PartitionedCallPartitionedCall7batch_normalization_58/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_136904Ч
+conv2d_transpose_31/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_55/PartitionedCall:output:0conv2d_transpose_31_136906conv2d_transpose_31_136908*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_31_layer_call_and_return_conditional_losses_136594Ђ
.batch_normalization_59/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_31/StatefulPartitionedCall:output:0batch_normalization_59_136911batch_normalization_59_136913batch_normalization_59_136915batch_normalization_59_136917*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_136623џ
leaky_re_lu_56/PartitionedCallPartitionedCall7batch_normalization_59/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_136925Ч
+conv2d_transpose_32/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_56/PartitionedCall:output:0conv2d_transpose_32_136927conv2d_transpose_32_136929*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_32_layer_call_and_return_conditional_losses_136703
IdentityIdentity4conv2d_transpose_32/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџЊ
NoOpNoOp/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_52/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall/^batch_normalization_54/StatefulPartitionedCall/^batch_normalization_55/StatefulPartitionedCall/^batch_normalization_56/StatefulPartitionedCall/^batch_normalization_57/StatefulPartitionedCall/^batch_normalization_58/StatefulPartitionedCall/^batch_normalization_59/StatefulPartitionedCall,^conv2d_transpose_22/StatefulPartitionedCall,^conv2d_transpose_23/StatefulPartitionedCall,^conv2d_transpose_24/StatefulPartitionedCall,^conv2d_transpose_25/StatefulPartitionedCall,^conv2d_transpose_26/StatefulPartitionedCall,^conv2d_transpose_27/StatefulPartitionedCall,^conv2d_transpose_28/StatefulPartitionedCall,^conv2d_transpose_29/StatefulPartitionedCall,^conv2d_transpose_30/StatefulPartitionedCall,^conv2d_transpose_31/StatefulPartitionedCall,^conv2d_transpose_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2`
.batch_normalization_54/StatefulPartitionedCall.batch_normalization_54/StatefulPartitionedCall2`
.batch_normalization_55/StatefulPartitionedCall.batch_normalization_55/StatefulPartitionedCall2`
.batch_normalization_56/StatefulPartitionedCall.batch_normalization_56/StatefulPartitionedCall2`
.batch_normalization_57/StatefulPartitionedCall.batch_normalization_57/StatefulPartitionedCall2`
.batch_normalization_58/StatefulPartitionedCall.batch_normalization_58/StatefulPartitionedCall2`
.batch_normalization_59/StatefulPartitionedCall.batch_normalization_59/StatefulPartitionedCall2Z
+conv2d_transpose_22/StatefulPartitionedCall+conv2d_transpose_22/StatefulPartitionedCall2Z
+conv2d_transpose_23/StatefulPartitionedCall+conv2d_transpose_23/StatefulPartitionedCall2Z
+conv2d_transpose_24/StatefulPartitionedCall+conv2d_transpose_24/StatefulPartitionedCall2Z
+conv2d_transpose_25/StatefulPartitionedCall+conv2d_transpose_25/StatefulPartitionedCall2Z
+conv2d_transpose_26/StatefulPartitionedCall+conv2d_transpose_26/StatefulPartitionedCall2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2Z
+conv2d_transpose_30/StatefulPartitionedCall+conv2d_transpose_30/StatefulPartitionedCall2Z
+conv2d_transpose_31/StatefulPartitionedCall+conv2d_transpose_31/StatefulPartitionedCall2Z
+conv2d_transpose_32/StatefulPartitionedCall+conv2d_transpose_32/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

С
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_136222

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Ц
D
!__inference__wrapped_model_135581
input_6_
Ddecoder_conv2d_transpose_22_conv2d_transpose_readvariableop_resource: I
;decoder_conv2d_transpose_22_biasadd_readvariableop_resource: D
6decoder_batch_normalization_50_readvariableop_resource: F
8decoder_batch_normalization_50_readvariableop_1_resource: U
Gdecoder_batch_normalization_50_fusedbatchnormv3_readvariableop_resource: W
Idecoder_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resource: ^
Ddecoder_conv2d_transpose_23_conv2d_transpose_readvariableop_resource:@ I
;decoder_conv2d_transpose_23_biasadd_readvariableop_resource:@D
6decoder_batch_normalization_51_readvariableop_resource:@F
8decoder_batch_normalization_51_readvariableop_1_resource:@U
Gdecoder_batch_normalization_51_fusedbatchnormv3_readvariableop_resource:@W
Idecoder_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resource:@_
Ddecoder_conv2d_transpose_24_conv2d_transpose_readvariableop_resource:@J
;decoder_conv2d_transpose_24_biasadd_readvariableop_resource:	E
6decoder_batch_normalization_52_readvariableop_resource:	G
8decoder_batch_normalization_52_readvariableop_1_resource:	V
Gdecoder_batch_normalization_52_fusedbatchnormv3_readvariableop_resource:	X
Idecoder_batch_normalization_52_fusedbatchnormv3_readvariableop_1_resource:	_
Ddecoder_conv2d_transpose_25_conv2d_transpose_readvariableop_resource:@I
;decoder_conv2d_transpose_25_biasadd_readvariableop_resource:@D
6decoder_batch_normalization_53_readvariableop_resource:@F
8decoder_batch_normalization_53_readvariableop_1_resource:@U
Gdecoder_batch_normalization_53_fusedbatchnormv3_readvariableop_resource:@W
Idecoder_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:@^
Ddecoder_conv2d_transpose_26_conv2d_transpose_readvariableop_resource:@@I
;decoder_conv2d_transpose_26_biasadd_readvariableop_resource:@D
6decoder_batch_normalization_54_readvariableop_resource:@F
8decoder_batch_normalization_54_readvariableop_1_resource:@U
Gdecoder_batch_normalization_54_fusedbatchnormv3_readvariableop_resource:@W
Idecoder_batch_normalization_54_fusedbatchnormv3_readvariableop_1_resource:@^
Ddecoder_conv2d_transpose_27_conv2d_transpose_readvariableop_resource: @I
;decoder_conv2d_transpose_27_biasadd_readvariableop_resource: D
6decoder_batch_normalization_55_readvariableop_resource: F
8decoder_batch_normalization_55_readvariableop_1_resource: U
Gdecoder_batch_normalization_55_fusedbatchnormv3_readvariableop_resource: W
Idecoder_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource: _
Ddecoder_conv2d_transpose_28_conv2d_transpose_readvariableop_resource: J
;decoder_conv2d_transpose_28_biasadd_readvariableop_resource:	E
6decoder_batch_normalization_56_readvariableop_resource:	G
8decoder_batch_normalization_56_readvariableop_1_resource:	V
Gdecoder_batch_normalization_56_fusedbatchnormv3_readvariableop_resource:	X
Idecoder_batch_normalization_56_fusedbatchnormv3_readvariableop_1_resource:	_
Ddecoder_conv2d_transpose_29_conv2d_transpose_readvariableop_resource:@I
;decoder_conv2d_transpose_29_biasadd_readvariableop_resource:@D
6decoder_batch_normalization_57_readvariableop_resource:@F
8decoder_batch_normalization_57_readvariableop_1_resource:@U
Gdecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_resource:@W
Idecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resource:@^
Ddecoder_conv2d_transpose_30_conv2d_transpose_readvariableop_resource: @I
;decoder_conv2d_transpose_30_biasadd_readvariableop_resource: D
6decoder_batch_normalization_58_readvariableop_resource: F
8decoder_batch_normalization_58_readvariableop_1_resource: U
Gdecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_resource: W
Idecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_1_resource: ^
Ddecoder_conv2d_transpose_31_conv2d_transpose_readvariableop_resource:  I
;decoder_conv2d_transpose_31_biasadd_readvariableop_resource: D
6decoder_batch_normalization_59_readvariableop_resource: F
8decoder_batch_normalization_59_readvariableop_1_resource: U
Gdecoder_batch_normalization_59_fusedbatchnormv3_readvariableop_resource: W
Idecoder_batch_normalization_59_fusedbatchnormv3_readvariableop_1_resource: ^
Ddecoder_conv2d_transpose_32_conv2d_transpose_readvariableop_resource: I
;decoder_conv2d_transpose_32_biasadd_readvariableop_resource:
identityЂ>decoder/batch_normalization_50/FusedBatchNormV3/ReadVariableOpЂ@decoder/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1Ђ-decoder/batch_normalization_50/ReadVariableOpЂ/decoder/batch_normalization_50/ReadVariableOp_1Ђ>decoder/batch_normalization_51/FusedBatchNormV3/ReadVariableOpЂ@decoder/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1Ђ-decoder/batch_normalization_51/ReadVariableOpЂ/decoder/batch_normalization_51/ReadVariableOp_1Ђ>decoder/batch_normalization_52/FusedBatchNormV3/ReadVariableOpЂ@decoder/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1Ђ-decoder/batch_normalization_52/ReadVariableOpЂ/decoder/batch_normalization_52/ReadVariableOp_1Ђ>decoder/batch_normalization_53/FusedBatchNormV3/ReadVariableOpЂ@decoder/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1Ђ-decoder/batch_normalization_53/ReadVariableOpЂ/decoder/batch_normalization_53/ReadVariableOp_1Ђ>decoder/batch_normalization_54/FusedBatchNormV3/ReadVariableOpЂ@decoder/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1Ђ-decoder/batch_normalization_54/ReadVariableOpЂ/decoder/batch_normalization_54/ReadVariableOp_1Ђ>decoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOpЂ@decoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1Ђ-decoder/batch_normalization_55/ReadVariableOpЂ/decoder/batch_normalization_55/ReadVariableOp_1Ђ>decoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOpЂ@decoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1Ђ-decoder/batch_normalization_56/ReadVariableOpЂ/decoder/batch_normalization_56/ReadVariableOp_1Ђ>decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOpЂ@decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1Ђ-decoder/batch_normalization_57/ReadVariableOpЂ/decoder/batch_normalization_57/ReadVariableOp_1Ђ>decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOpЂ@decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1Ђ-decoder/batch_normalization_58/ReadVariableOpЂ/decoder/batch_normalization_58/ReadVariableOp_1Ђ>decoder/batch_normalization_59/FusedBatchNormV3/ReadVariableOpЂ@decoder/batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1Ђ-decoder/batch_normalization_59/ReadVariableOpЂ/decoder/batch_normalization_59/ReadVariableOp_1Ђ2decoder/conv2d_transpose_22/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_22/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_23/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_23/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_24/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_24/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_25/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_25/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_26/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_26/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_27/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_28/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_29/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_29/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_30/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_30/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_31/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_31/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_32/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_32/conv2d_transpose/ReadVariableOpX
!decoder/conv2d_transpose_22/ShapeShapeinput_6*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_22/strided_sliceStridedSlice*decoder/conv2d_transpose_22/Shape:output:08decoder/conv2d_transpose_22/strided_slice/stack:output:0:decoder/conv2d_transpose_22/strided_slice/stack_1:output:0:decoder/conv2d_transpose_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_22/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_22/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_22/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
!decoder/conv2d_transpose_22/stackPack2decoder/conv2d_transpose_22/strided_slice:output:0,decoder/conv2d_transpose_22/stack/1:output:0,decoder/conv2d_transpose_22/stack/2:output:0,decoder/conv2d_transpose_22/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_22/strided_slice_1StridedSlice*decoder/conv2d_transpose_22/stack:output:0:decoder/conv2d_transpose_22/strided_slice_1/stack:output:0<decoder/conv2d_transpose_22/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
;decoder/conv2d_transpose_22/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_22_conv2d_transpose_readvariableop_resource*'
_output_shapes
: *
dtype0 
,decoder/conv2d_transpose_22/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_22/stack:output:0Cdecoder/conv2d_transpose_22/conv2d_transpose/ReadVariableOp:value:0input_6*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
Њ
2decoder/conv2d_transpose_22/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0л
#decoder/conv2d_transpose_22/BiasAddBiasAdd5decoder/conv2d_transpose_22/conv2d_transpose:output:0:decoder/conv2d_transpose_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  
-decoder/batch_normalization_50/ReadVariableOpReadVariableOp6decoder_batch_normalization_50_readvariableop_resource*
_output_shapes
: *
dtype0Є
/decoder/batch_normalization_50/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_50_readvariableop_1_resource*
_output_shapes
: *
dtype0Т
>decoder/batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ц
@decoder/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ї
/decoder/batch_normalization_50/FusedBatchNormV3FusedBatchNormV3,decoder/conv2d_transpose_22/BiasAdd:output:05decoder/batch_normalization_50/ReadVariableOp:value:07decoder/batch_normalization_50/ReadVariableOp_1:value:0Fdecoder/batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0Hdecoder/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( Ѓ
 decoder/leaky_re_lu_47/LeakyRelu	LeakyRelu3decoder/batch_normalization_50/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ *
alpha%>
!decoder/conv2d_transpose_23/ShapeShape.decoder/leaky_re_lu_47/LeakyRelu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_23/strided_sliceStridedSlice*decoder/conv2d_transpose_23/Shape:output:08decoder/conv2d_transpose_23/strided_slice/stack:output:0:decoder/conv2d_transpose_23/strided_slice/stack_1:output:0:decoder/conv2d_transpose_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_23/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_23/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_23/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@
!decoder/conv2d_transpose_23/stackPack2decoder/conv2d_transpose_23/strided_slice:output:0,decoder/conv2d_transpose_23/stack/1:output:0,decoder/conv2d_transpose_23/stack/2:output:0,decoder/conv2d_transpose_23/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_23/strided_slice_1StridedSlice*decoder/conv2d_transpose_23/stack:output:0:decoder/conv2d_transpose_23/strided_slice_1/stack:output:0<decoder/conv2d_transpose_23/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
;decoder/conv2d_transpose_23/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0Ц
,decoder/conv2d_transpose_23/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_23/stack:output:0Cdecoder/conv2d_transpose_23/conv2d_transpose/ReadVariableOp:value:0.decoder/leaky_re_lu_47/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_23/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0л
#decoder/conv2d_transpose_23/BiasAddBiasAdd5decoder/conv2d_transpose_23/conv2d_transpose:output:0:decoder/conv2d_transpose_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 
-decoder/batch_normalization_51/ReadVariableOpReadVariableOp6decoder_batch_normalization_51_readvariableop_resource*
_output_shapes
:@*
dtype0Є
/decoder/batch_normalization_51/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_51_readvariableop_1_resource*
_output_shapes
:@*
dtype0Т
>decoder/batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ц
@decoder/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ї
/decoder/batch_normalization_51/FusedBatchNormV3FusedBatchNormV3,decoder/conv2d_transpose_23/BiasAdd:output:05decoder/batch_normalization_51/ReadVariableOp:value:07decoder/batch_normalization_51/ReadVariableOp_1:value:0Fdecoder/batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0Hdecoder/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( Ѓ
 decoder/leaky_re_lu_48/LeakyRelu	LeakyRelu3decoder/batch_normalization_51/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@*
alpha%>
!decoder/conv2d_transpose_24/ShapeShape.decoder/leaky_re_lu_48/LeakyRelu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_24/strided_sliceStridedSlice*decoder/conv2d_transpose_24/Shape:output:08decoder/conv2d_transpose_24/strided_slice/stack:output:0:decoder/conv2d_transpose_24/strided_slice/stack_1:output:0:decoder/conv2d_transpose_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_24/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_24/stack/2Const*
_output_shapes
: *
dtype0*
value	B :f
#decoder/conv2d_transpose_24/stack/3Const*
_output_shapes
: *
dtype0*
value
B :
!decoder/conv2d_transpose_24/stackPack2decoder/conv2d_transpose_24/strided_slice:output:0,decoder/conv2d_transpose_24/stack/1:output:0,decoder/conv2d_transpose_24/stack/2:output:0,decoder/conv2d_transpose_24/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_24/strided_slice_1StridedSlice*decoder/conv2d_transpose_24/stack:output:0:decoder/conv2d_transpose_24/strided_slice_1/stack:output:0<decoder/conv2d_transpose_24/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
;decoder/conv2d_transpose_24/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_24_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0Ч
,decoder/conv2d_transpose_24/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_24/stack:output:0Cdecoder/conv2d_transpose_24/conv2d_transpose/ReadVariableOp:value:0.decoder/leaky_re_lu_48/LeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Ћ
2decoder/conv2d_transpose_24/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0м
#decoder/conv2d_transpose_24/BiasAddBiasAdd5decoder/conv2d_transpose_24/conv2d_transpose:output:0:decoder/conv2d_transpose_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЁ
-decoder/batch_normalization_52/ReadVariableOpReadVariableOp6decoder_batch_normalization_52_readvariableop_resource*
_output_shapes	
:*
dtype0Ѕ
/decoder/batch_normalization_52/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_52_readvariableop_1_resource*
_output_shapes	
:*
dtype0У
>decoder/batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ч
@decoder/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ќ
/decoder/batch_normalization_52/FusedBatchNormV3FusedBatchNormV3,decoder/conv2d_transpose_24/BiasAdd:output:05decoder/batch_normalization_52/ReadVariableOp:value:07decoder/batch_normalization_52/ReadVariableOp_1:value:0Fdecoder/batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0Hdecoder/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Є
 decoder/leaky_re_lu_49/LeakyRelu	LeakyRelu3decoder/batch_normalization_52/FusedBatchNormV3:y:0*0
_output_shapes
:џџџџџџџџџ*
alpha%>
!decoder/conv2d_transpose_25/ShapeShape.decoder/leaky_re_lu_49/LeakyRelu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_25/strided_sliceStridedSlice*decoder/conv2d_transpose_25/Shape:output:08decoder/conv2d_transpose_25/strided_slice/stack:output:0:decoder/conv2d_transpose_25/strided_slice/stack_1:output:0:decoder/conv2d_transpose_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_25/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_25/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_25/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@
!decoder/conv2d_transpose_25/stackPack2decoder/conv2d_transpose_25/strided_slice:output:0,decoder/conv2d_transpose_25/stack/1:output:0,decoder/conv2d_transpose_25/stack/2:output:0,decoder/conv2d_transpose_25/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_25/strided_slice_1StridedSlice*decoder/conv2d_transpose_25/stack:output:0:decoder/conv2d_transpose_25/strided_slice_1/stack:output:0<decoder/conv2d_transpose_25/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
;decoder/conv2d_transpose_25/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_25_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0Ц
,decoder/conv2d_transpose_25/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_25/stack:output:0Cdecoder/conv2d_transpose_25/conv2d_transpose/ReadVariableOp:value:0.decoder/leaky_re_lu_49/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_25/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0л
#decoder/conv2d_transpose_25/BiasAddBiasAdd5decoder/conv2d_transpose_25/conv2d_transpose:output:0:decoder/conv2d_transpose_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 
-decoder/batch_normalization_53/ReadVariableOpReadVariableOp6decoder_batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype0Є
/decoder/batch_normalization_53/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype0Т
>decoder/batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ц
@decoder/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ї
/decoder/batch_normalization_53/FusedBatchNormV3FusedBatchNormV3,decoder/conv2d_transpose_25/BiasAdd:output:05decoder/batch_normalization_53/ReadVariableOp:value:07decoder/batch_normalization_53/ReadVariableOp_1:value:0Fdecoder/batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0Hdecoder/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( Ѓ
 decoder/leaky_re_lu_50/LeakyRelu	LeakyRelu3decoder/batch_normalization_53/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@*
alpha%>
!decoder/conv2d_transpose_26/ShapeShape.decoder/leaky_re_lu_50/LeakyRelu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_26/strided_sliceStridedSlice*decoder/conv2d_transpose_26/Shape:output:08decoder/conv2d_transpose_26/strided_slice/stack:output:0:decoder/conv2d_transpose_26/strided_slice/stack_1:output:0:decoder/conv2d_transpose_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_26/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_26/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_26/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@
!decoder/conv2d_transpose_26/stackPack2decoder/conv2d_transpose_26/strided_slice:output:0,decoder/conv2d_transpose_26/stack/1:output:0,decoder/conv2d_transpose_26/stack/2:output:0,decoder/conv2d_transpose_26/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_26/strided_slice_1StridedSlice*decoder/conv2d_transpose_26/stack:output:0:decoder/conv2d_transpose_26/strided_slice_1/stack:output:0<decoder/conv2d_transpose_26/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
;decoder/conv2d_transpose_26/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_26_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ц
,decoder/conv2d_transpose_26/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_26/stack:output:0Cdecoder/conv2d_transpose_26/conv2d_transpose/ReadVariableOp:value:0.decoder/leaky_re_lu_50/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_26/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0л
#decoder/conv2d_transpose_26/BiasAddBiasAdd5decoder/conv2d_transpose_26/conv2d_transpose:output:0:decoder/conv2d_transpose_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ 
-decoder/batch_normalization_54/ReadVariableOpReadVariableOp6decoder_batch_normalization_54_readvariableop_resource*
_output_shapes
:@*
dtype0Є
/decoder/batch_normalization_54/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_54_readvariableop_1_resource*
_output_shapes
:@*
dtype0Т
>decoder/batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ц
@decoder/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ї
/decoder/batch_normalization_54/FusedBatchNormV3FusedBatchNormV3,decoder/conv2d_transpose_26/BiasAdd:output:05decoder/batch_normalization_54/ReadVariableOp:value:07decoder/batch_normalization_54/ReadVariableOp_1:value:0Fdecoder/batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0Hdecoder/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( Ѓ
 decoder/leaky_re_lu_51/LeakyRelu	LeakyRelu3decoder/batch_normalization_54/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@*
alpha%>
!decoder/conv2d_transpose_27/ShapeShape.decoder/leaky_re_lu_51/LeakyRelu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_27/strided_sliceStridedSlice*decoder/conv2d_transpose_27/Shape:output:08decoder/conv2d_transpose_27/strided_slice/stack:output:0:decoder/conv2d_transpose_27/strided_slice/stack_1:output:0:decoder/conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
!decoder/conv2d_transpose_27/stackPack2decoder/conv2d_transpose_27/strided_slice:output:0,decoder/conv2d_transpose_27/stack/1:output:0,decoder/conv2d_transpose_27/stack/2:output:0,decoder/conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_27/strided_slice_1StridedSlice*decoder/conv2d_transpose_27/stack:output:0:decoder/conv2d_transpose_27/strided_slice_1/stack:output:0<decoder/conv2d_transpose_27/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
;decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0Ц
,decoder/conv2d_transpose_27/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_27/stack:output:0Cdecoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0.decoder/leaky_re_lu_51/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0л
#decoder/conv2d_transpose_27/BiasAddBiasAdd5decoder/conv2d_transpose_27/conv2d_transpose:output:0:decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  
-decoder/batch_normalization_55/ReadVariableOpReadVariableOp6decoder_batch_normalization_55_readvariableop_resource*
_output_shapes
: *
dtype0Є
/decoder/batch_normalization_55/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_55_readvariableop_1_resource*
_output_shapes
: *
dtype0Т
>decoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ц
@decoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ї
/decoder/batch_normalization_55/FusedBatchNormV3FusedBatchNormV3,decoder/conv2d_transpose_27/BiasAdd:output:05decoder/batch_normalization_55/ReadVariableOp:value:07decoder/batch_normalization_55/ReadVariableOp_1:value:0Fdecoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0Hdecoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( Ѓ
 decoder/leaky_re_lu_52/LeakyRelu	LeakyRelu3decoder/batch_normalization_55/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ *
alpha%>
!decoder/conv2d_transpose_28/ShapeShape.decoder/leaky_re_lu_52/LeakyRelu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_28/strided_sliceStridedSlice*decoder/conv2d_transpose_28/Shape:output:08decoder/conv2d_transpose_28/strided_slice/stack:output:0:decoder/conv2d_transpose_28/strided_slice/stack_1:output:0:decoder/conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B : e
#decoder/conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value	B : f
#decoder/conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value
B :
!decoder/conv2d_transpose_28/stackPack2decoder/conv2d_transpose_28/strided_slice:output:0,decoder/conv2d_transpose_28/stack/1:output:0,decoder/conv2d_transpose_28/stack/2:output:0,decoder/conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_28/strided_slice_1StridedSlice*decoder/conv2d_transpose_28/stack:output:0:decoder/conv2d_transpose_28/strided_slice_1/stack:output:0<decoder/conv2d_transpose_28/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
;decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_28_conv2d_transpose_readvariableop_resource*'
_output_shapes
: *
dtype0Ч
,decoder/conv2d_transpose_28/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_28/stack:output:0Cdecoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0.decoder/leaky_re_lu_52/LeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides
Ћ
2decoder/conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0м
#decoder/conv2d_transpose_28/BiasAddBiasAdd5decoder/conv2d_transpose_28/conv2d_transpose:output:0:decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  Ё
-decoder/batch_normalization_56/ReadVariableOpReadVariableOp6decoder_batch_normalization_56_readvariableop_resource*
_output_shapes	
:*
dtype0Ѕ
/decoder/batch_normalization_56/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_56_readvariableop_1_resource*
_output_shapes	
:*
dtype0У
>decoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_56_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ч
@decoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_56_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ќ
/decoder/batch_normalization_56/FusedBatchNormV3FusedBatchNormV3,decoder/conv2d_transpose_28/BiasAdd:output:05decoder/batch_normalization_56/ReadVariableOp:value:07decoder/batch_normalization_56/ReadVariableOp_1:value:0Fdecoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp:value:0Hdecoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ  :::::*
epsilon%o:*
is_training( Є
 decoder/leaky_re_lu_53/LeakyRelu	LeakyRelu3decoder/batch_normalization_56/FusedBatchNormV3:y:0*0
_output_shapes
:џџџџџџџџџ  *
alpha%>
!decoder/conv2d_transpose_29/ShapeShape.decoder/leaky_re_lu_53/LeakyRelu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_29/strided_sliceStridedSlice*decoder/conv2d_transpose_29/Shape:output:08decoder/conv2d_transpose_29/strided_slice/stack:output:0:decoder/conv2d_transpose_29/strided_slice/stack_1:output:0:decoder/conv2d_transpose_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_29/stack/1Const*
_output_shapes
: *
dtype0*
value	B : e
#decoder/conv2d_transpose_29/stack/2Const*
_output_shapes
: *
dtype0*
value	B : e
#decoder/conv2d_transpose_29/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@
!decoder/conv2d_transpose_29/stackPack2decoder/conv2d_transpose_29/strided_slice:output:0,decoder/conv2d_transpose_29/stack/1:output:0,decoder/conv2d_transpose_29/stack/2:output:0,decoder/conv2d_transpose_29/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_29/strided_slice_1StridedSlice*decoder/conv2d_transpose_29/stack:output:0:decoder/conv2d_transpose_29/strided_slice_1/stack:output:0<decoder/conv2d_transpose_29/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
;decoder/conv2d_transpose_29/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_29_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0Ц
,decoder/conv2d_transpose_29/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_29/stack:output:0Cdecoder/conv2d_transpose_29/conv2d_transpose/ReadVariableOp:value:0.decoder/leaky_re_lu_53/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  @*
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_29/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0л
#decoder/conv2d_transpose_29/BiasAddBiasAdd5decoder/conv2d_transpose_29/conv2d_transpose:output:0:decoder/conv2d_transpose_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  @ 
-decoder/batch_normalization_57/ReadVariableOpReadVariableOp6decoder_batch_normalization_57_readvariableop_resource*
_output_shapes
:@*
dtype0Є
/decoder/batch_normalization_57/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_57_readvariableop_1_resource*
_output_shapes
:@*
dtype0Т
>decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ц
@decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_57_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ї
/decoder/batch_normalization_57/FusedBatchNormV3FusedBatchNormV3,decoder/conv2d_transpose_29/BiasAdd:output:05decoder/batch_normalization_57/ReadVariableOp:value:07decoder/batch_normalization_57/ReadVariableOp_1:value:0Fdecoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp:value:0Hdecoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ  @:@:@:@:@:*
epsilon%o:*
is_training( Ѓ
 decoder/leaky_re_lu_54/LeakyRelu	LeakyRelu3decoder/batch_normalization_57/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ  @*
alpha%>
!decoder/conv2d_transpose_30/ShapeShape.decoder/leaky_re_lu_54/LeakyRelu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_30/strided_sliceStridedSlice*decoder/conv2d_transpose_30/Shape:output:08decoder/conv2d_transpose_30/strided_slice/stack:output:0:decoder/conv2d_transpose_30/strided_slice/stack_1:output:0:decoder/conv2d_transpose_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_30/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@e
#decoder/conv2d_transpose_30/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@e
#decoder/conv2d_transpose_30/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
!decoder/conv2d_transpose_30/stackPack2decoder/conv2d_transpose_30/strided_slice:output:0,decoder/conv2d_transpose_30/stack/1:output:0,decoder/conv2d_transpose_30/stack/2:output:0,decoder/conv2d_transpose_30/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_30/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_30/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_30/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_30/strided_slice_1StridedSlice*decoder/conv2d_transpose_30/stack:output:0:decoder/conv2d_transpose_30/strided_slice_1/stack:output:0<decoder/conv2d_transpose_30/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_30/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
;decoder/conv2d_transpose_30/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_30_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0Ц
,decoder/conv2d_transpose_30/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_30/stack:output:0Cdecoder/conv2d_transpose_30/conv2d_transpose/ReadVariableOp:value:0.decoder/leaky_re_lu_54/LeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@ *
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_30/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_30_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0л
#decoder/conv2d_transpose_30/BiasAddBiasAdd5decoder/conv2d_transpose_30/conv2d_transpose:output:0:decoder/conv2d_transpose_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@@  
-decoder/batch_normalization_58/ReadVariableOpReadVariableOp6decoder_batch_normalization_58_readvariableop_resource*
_output_shapes
: *
dtype0Є
/decoder/batch_normalization_58/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_58_readvariableop_1_resource*
_output_shapes
: *
dtype0Т
>decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ц
@decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_58_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ї
/decoder/batch_normalization_58/FusedBatchNormV3FusedBatchNormV3,decoder/conv2d_transpose_30/BiasAdd:output:05decoder/batch_normalization_58/ReadVariableOp:value:07decoder/batch_normalization_58/ReadVariableOp_1:value:0Fdecoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp:value:0Hdecoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@@ : : : : :*
epsilon%o:*
is_training( Ѓ
 decoder/leaky_re_lu_55/LeakyRelu	LeakyRelu3decoder/batch_normalization_58/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ@@ *
alpha%>
!decoder/conv2d_transpose_31/ShapeShape.decoder/leaky_re_lu_55/LeakyRelu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_31/strided_sliceStridedSlice*decoder/conv2d_transpose_31/Shape:output:08decoder/conv2d_transpose_31/strided_slice/stack:output:0:decoder/conv2d_transpose_31/strided_slice/stack_1:output:0:decoder/conv2d_transpose_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#decoder/conv2d_transpose_31/stack/1Const*
_output_shapes
: *
dtype0*
value
B :f
#decoder/conv2d_transpose_31/stack/2Const*
_output_shapes
: *
dtype0*
value
B :e
#decoder/conv2d_transpose_31/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
!decoder/conv2d_transpose_31/stackPack2decoder/conv2d_transpose_31/strided_slice:output:0,decoder/conv2d_transpose_31/stack/1:output:0,decoder/conv2d_transpose_31/stack/2:output:0,decoder/conv2d_transpose_31/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_31/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_31/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_31/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_31/strided_slice_1StridedSlice*decoder/conv2d_transpose_31/stack:output:0:decoder/conv2d_transpose_31/strided_slice_1/stack:output:0<decoder/conv2d_transpose_31/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_31/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
;decoder/conv2d_transpose_31/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_31_conv2d_transpose_readvariableop_resource*&
_output_shapes
:  *
dtype0Ш
,decoder/conv2d_transpose_31/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_31/stack:output:0Cdecoder/conv2d_transpose_31/conv2d_transpose/ReadVariableOp:value:0.decoder/leaky_re_lu_55/LeakyRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_31/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0н
#decoder/conv2d_transpose_31/BiasAddBiasAdd5decoder/conv2d_transpose_31/conv2d_transpose:output:0:decoder/conv2d_transpose_31/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ  
-decoder/batch_normalization_59/ReadVariableOpReadVariableOp6decoder_batch_normalization_59_readvariableop_resource*
_output_shapes
: *
dtype0Є
/decoder/batch_normalization_59/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_59_readvariableop_1_resource*
_output_shapes
: *
dtype0Т
>decoder/batch_normalization_59/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_59_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ц
@decoder/batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_59_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0љ
/decoder/batch_normalization_59/FusedBatchNormV3FusedBatchNormV3,decoder/conv2d_transpose_31/BiasAdd:output:05decoder/batch_normalization_59/ReadVariableOp:value:07decoder/batch_normalization_59/ReadVariableOp_1:value:0Fdecoder/batch_normalization_59/FusedBatchNormV3/ReadVariableOp:value:0Hdecoder/batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( Ѕ
 decoder/leaky_re_lu_56/LeakyRelu	LeakyRelu3decoder/batch_normalization_59/FusedBatchNormV3:y:0*1
_output_shapes
:џџџџџџџџџ *
alpha%>
!decoder/conv2d_transpose_32/ShapeShape.decoder/leaky_re_lu_56/LeakyRelu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_32/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_32/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_32/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_32/strided_sliceStridedSlice*decoder/conv2d_transpose_32/Shape:output:08decoder/conv2d_transpose_32/strided_slice/stack:output:0:decoder/conv2d_transpose_32/strided_slice/stack_1:output:0:decoder/conv2d_transpose_32/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#decoder/conv2d_transpose_32/stack/1Const*
_output_shapes
: *
dtype0*
value
B :f
#decoder/conv2d_transpose_32/stack/2Const*
_output_shapes
: *
dtype0*
value
B :e
#decoder/conv2d_transpose_32/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
!decoder/conv2d_transpose_32/stackPack2decoder/conv2d_transpose_32/strided_slice:output:0,decoder/conv2d_transpose_32/stack/1:output:0,decoder/conv2d_transpose_32/stack/2:output:0,decoder/conv2d_transpose_32/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_32/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_32/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_32/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_32/strided_slice_1StridedSlice*decoder/conv2d_transpose_32/stack:output:0:decoder/conv2d_transpose_32/strided_slice_1/stack:output:0<decoder/conv2d_transpose_32/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_32/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
;decoder/conv2d_transpose_32/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_32_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ш
,decoder/conv2d_transpose_32/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_32/stack:output:0Cdecoder/conv2d_transpose_32/conv2d_transpose/ReadVariableOp:value:0.decoder/leaky_re_lu_56/LeakyRelu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_32/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0н
#decoder/conv2d_transpose_32/BiasAddBiasAdd5decoder/conv2d_transpose_32/conv2d_transpose:output:0:decoder/conv2d_transpose_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџ
#decoder/conv2d_transpose_32/SigmoidSigmoid,decoder/conv2d_transpose_32/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџ
IdentityIdentity'decoder/conv2d_transpose_32/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџГ
NoOpNoOp?^decoder/batch_normalization_50/FusedBatchNormV3/ReadVariableOpA^decoder/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1.^decoder/batch_normalization_50/ReadVariableOp0^decoder/batch_normalization_50/ReadVariableOp_1?^decoder/batch_normalization_51/FusedBatchNormV3/ReadVariableOpA^decoder/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1.^decoder/batch_normalization_51/ReadVariableOp0^decoder/batch_normalization_51/ReadVariableOp_1?^decoder/batch_normalization_52/FusedBatchNormV3/ReadVariableOpA^decoder/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1.^decoder/batch_normalization_52/ReadVariableOp0^decoder/batch_normalization_52/ReadVariableOp_1?^decoder/batch_normalization_53/FusedBatchNormV3/ReadVariableOpA^decoder/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1.^decoder/batch_normalization_53/ReadVariableOp0^decoder/batch_normalization_53/ReadVariableOp_1?^decoder/batch_normalization_54/FusedBatchNormV3/ReadVariableOpA^decoder/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1.^decoder/batch_normalization_54/ReadVariableOp0^decoder/batch_normalization_54/ReadVariableOp_1?^decoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOpA^decoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1.^decoder/batch_normalization_55/ReadVariableOp0^decoder/batch_normalization_55/ReadVariableOp_1?^decoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOpA^decoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1.^decoder/batch_normalization_56/ReadVariableOp0^decoder/batch_normalization_56/ReadVariableOp_1?^decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOpA^decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1.^decoder/batch_normalization_57/ReadVariableOp0^decoder/batch_normalization_57/ReadVariableOp_1?^decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOpA^decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1.^decoder/batch_normalization_58/ReadVariableOp0^decoder/batch_normalization_58/ReadVariableOp_1?^decoder/batch_normalization_59/FusedBatchNormV3/ReadVariableOpA^decoder/batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1.^decoder/batch_normalization_59/ReadVariableOp0^decoder/batch_normalization_59/ReadVariableOp_13^decoder/conv2d_transpose_22/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_22/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_23/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_23/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_24/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_24/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_25/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_25/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_26/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_26/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_29/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_29/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_30/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_30/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_31/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_31/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_32/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_32/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
>decoder/batch_normalization_50/FusedBatchNormV3/ReadVariableOp>decoder/batch_normalization_50/FusedBatchNormV3/ReadVariableOp2
@decoder/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1@decoder/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12^
-decoder/batch_normalization_50/ReadVariableOp-decoder/batch_normalization_50/ReadVariableOp2b
/decoder/batch_normalization_50/ReadVariableOp_1/decoder/batch_normalization_50/ReadVariableOp_12
>decoder/batch_normalization_51/FusedBatchNormV3/ReadVariableOp>decoder/batch_normalization_51/FusedBatchNormV3/ReadVariableOp2
@decoder/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1@decoder/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12^
-decoder/batch_normalization_51/ReadVariableOp-decoder/batch_normalization_51/ReadVariableOp2b
/decoder/batch_normalization_51/ReadVariableOp_1/decoder/batch_normalization_51/ReadVariableOp_12
>decoder/batch_normalization_52/FusedBatchNormV3/ReadVariableOp>decoder/batch_normalization_52/FusedBatchNormV3/ReadVariableOp2
@decoder/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1@decoder/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12^
-decoder/batch_normalization_52/ReadVariableOp-decoder/batch_normalization_52/ReadVariableOp2b
/decoder/batch_normalization_52/ReadVariableOp_1/decoder/batch_normalization_52/ReadVariableOp_12
>decoder/batch_normalization_53/FusedBatchNormV3/ReadVariableOp>decoder/batch_normalization_53/FusedBatchNormV3/ReadVariableOp2
@decoder/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1@decoder/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12^
-decoder/batch_normalization_53/ReadVariableOp-decoder/batch_normalization_53/ReadVariableOp2b
/decoder/batch_normalization_53/ReadVariableOp_1/decoder/batch_normalization_53/ReadVariableOp_12
>decoder/batch_normalization_54/FusedBatchNormV3/ReadVariableOp>decoder/batch_normalization_54/FusedBatchNormV3/ReadVariableOp2
@decoder/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1@decoder/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12^
-decoder/batch_normalization_54/ReadVariableOp-decoder/batch_normalization_54/ReadVariableOp2b
/decoder/batch_normalization_54/ReadVariableOp_1/decoder/batch_normalization_54/ReadVariableOp_12
>decoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp>decoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp2
@decoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1@decoder/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12^
-decoder/batch_normalization_55/ReadVariableOp-decoder/batch_normalization_55/ReadVariableOp2b
/decoder/batch_normalization_55/ReadVariableOp_1/decoder/batch_normalization_55/ReadVariableOp_12
>decoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp>decoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp2
@decoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_1@decoder/batch_normalization_56/FusedBatchNormV3/ReadVariableOp_12^
-decoder/batch_normalization_56/ReadVariableOp-decoder/batch_normalization_56/ReadVariableOp2b
/decoder/batch_normalization_56/ReadVariableOp_1/decoder/batch_normalization_56/ReadVariableOp_12
>decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp>decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp2
@decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_1@decoder/batch_normalization_57/FusedBatchNormV3/ReadVariableOp_12^
-decoder/batch_normalization_57/ReadVariableOp-decoder/batch_normalization_57/ReadVariableOp2b
/decoder/batch_normalization_57/ReadVariableOp_1/decoder/batch_normalization_57/ReadVariableOp_12
>decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp>decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp2
@decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_1@decoder/batch_normalization_58/FusedBatchNormV3/ReadVariableOp_12^
-decoder/batch_normalization_58/ReadVariableOp-decoder/batch_normalization_58/ReadVariableOp2b
/decoder/batch_normalization_58/ReadVariableOp_1/decoder/batch_normalization_58/ReadVariableOp_12
>decoder/batch_normalization_59/FusedBatchNormV3/ReadVariableOp>decoder/batch_normalization_59/FusedBatchNormV3/ReadVariableOp2
@decoder/batch_normalization_59/FusedBatchNormV3/ReadVariableOp_1@decoder/batch_normalization_59/FusedBatchNormV3/ReadVariableOp_12^
-decoder/batch_normalization_59/ReadVariableOp-decoder/batch_normalization_59/ReadVariableOp2b
/decoder/batch_normalization_59/ReadVariableOp_1/decoder/batch_normalization_59/ReadVariableOp_12h
2decoder/conv2d_transpose_22/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_22/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_22/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_22/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_23/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_23/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_23/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_23/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_24/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_24/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_24/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_24/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_25/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_25/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_25/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_25/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_26/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_26/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_26/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_26/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_27/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_27/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_28/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_28/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_29/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_29/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_29/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_29/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_30/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_30/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_30/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_30/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_31/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_31/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_31/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_31/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_32/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_32/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_32/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_32/conv2d_transpose/ReadVariableOp:Y U
0
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6

Х
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_135898

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
н
Ё
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_135867

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
­
Д
(__inference_decoder_layer_call_fn_137666
input_6"
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:@ 
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29: @

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: %

unknown_35: 

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	

unknown_40:	%

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47: @

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: $

unknown_53:  

unknown_54: 

unknown_55: 

unknown_56: 

unknown_57: 

unknown_58: $

unknown_59: 

unknown_60:
identityЂStatefulPartitionedCall	
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ*L
_read_only_resource_inputs.
,*	
 !"%&'(+,-.1234789:=>*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_137410y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
	
в
7__inference_batch_normalization_50_layer_call_fn_139160

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_135651
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Щ
Љ
4__inference_conv2d_transpose_32_layer_call_fn_140254

inputs!
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_32_layer_call_and_return_conditional_losses_136703
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_48_layer_call_and_return_conditional_losses_136757

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ@*
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_139561

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ@*
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_136904

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ@@ *
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@@ :W S
/
_output_shapes
:џџџџџџџџџ@@ 
 
_user_specified_nameinputs
л 

O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_139945

inputsC
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_136778

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:џџџџџџџџџ*
alpha%>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_58_layer_call_fn_140072

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_136515
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Э
K
/__inference_leaky_re_lu_53_layer_call_fn_139898

inputs
identityО
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_136862i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ  :X T
0
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
з 

O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_139261

inputsB
(conv2d_transpose_readvariableop_resource:@ -
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂconv2d_transpose/ReadVariableOp;
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
valueB:б
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
valueB:й
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
valueB:й
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
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
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
valueB:й
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype0м
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_58_layer_call_fn_140085

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_136546
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
	
в
7__inference_batch_normalization_54_layer_call_fn_139629

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_136114
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
О
Г
(__inference_decoder_layer_call_fn_138244

inputs"
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:@ 
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@$

unknown_29: @

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: 

unknown_34: %

unknown_35: 

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:	

unknown_40:	%

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@

unknown_45:@

unknown_46:@$

unknown_47: @

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: $

unknown_53:  

unknown_54: 

unknown_55: 

unknown_56: 

unknown_57: 

unknown_58: $

unknown_59: 

unknown_60:
identityЂStatefulPartitionedCallЁ	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџ*`
_read_only_resource_inputsB
@>	
 !"#$%&'()*+,-./0123456789:;<=>*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_136933y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
Љ
4__inference_conv2d_transpose_23_layer_call_fn_139228

inputs!
unknown:@ 
	unknown_0:@
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_135730
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_136841

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:џџџџџџџџџ *
alpha%>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
н
Ё
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_136299

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Щ
K
/__inference_leaky_re_lu_48_layer_call_fn_139328

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_leaky_re_lu_48_layer_call_and_return_conditional_losses_136757h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

f
J__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_139447

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:џџџџџџџџџ*
alpha%>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Щ
serving_defaultЕ
D
input_69
serving_default_input_6:0џџџџџџџџџQ
conv2d_transpose_32:
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:њ
З	
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
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
layer_with_weights-17
layer-26
layer-27
layer_with_weights-18
layer-28
layer_with_weights-19
layer-29
layer-30
 layer_with_weights-20
 layer-31
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_default_save_signature
(
signatures"
_tf_keras_network
"
_tf_keras_input_layer
н
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
ъ
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8axis
	9gamma
:beta
;moving_mean
<moving_variance"
_tf_keras_layer
Ѕ
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
н
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias
 K_jit_compiled_convolution_op"
_tf_keras_layer
ъ
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Raxis
	Sgamma
Tbeta
Umoving_mean
Vmoving_variance"
_tf_keras_layer
Ѕ
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
н
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias
 e_jit_compiled_convolution_op"
_tf_keras_layer
ъ
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance"
_tf_keras_layer
Ѕ
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
н
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

}kernel
~bias
 _jit_compiled_convolution_op"
_tf_keras_layer
ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	 axis

Ёgamma
	Ђbeta
Ѓmoving_mean
Єmoving_variance"
_tf_keras_layer
Ћ
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Ј	keras_api
Љ__call__
+Њ&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Ћ	variables
Ќtrainable_variables
­regularization_losses
Ў	keras_api
Џ__call__
+А&call_and_return_all_conditional_losses
Бkernel
	Вbias
!Г_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses
	Кaxis

Лgamma
	Мbeta
Нmoving_mean
Оmoving_variance"
_tf_keras_layer
Ћ
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses
Ыkernel
	Ьbias
!Э_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
Ю	variables
Яtrainable_variables
аregularization_losses
б	keras_api
в__call__
+г&call_and_return_all_conditional_losses
	дaxis

еgamma
	жbeta
зmoving_mean
иmoving_variance"
_tf_keras_layer
Ћ
й	variables
кtrainable_variables
лregularization_losses
м	keras_api
н__call__
+о&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
п	variables
рtrainable_variables
сregularization_losses
т	keras_api
у__call__
+ф&call_and_return_all_conditional_losses
хkernel
	цbias
!ч_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses
	юaxis

яgamma
	№beta
ёmoving_mean
ђmoving_variance"
_tf_keras_layer
Ћ
ѓ	variables
єtrainable_variables
ѕregularization_losses
і	keras_api
ї__call__
+ј&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
љ	variables
њtrainable_variables
ћregularization_losses
ќ	keras_api
§__call__
+ў&call_and_return_all_conditional_losses
џkernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ѕ
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+Ё&call_and_return_all_conditional_losses
	Ђaxis

Ѓgamma
	Єbeta
Ѕmoving_mean
Іmoving_variance"
_tf_keras_layer
Ћ
Ї	variables
Јtrainable_variables
Љregularization_losses
Њ	keras_api
Ћ__call__
+Ќ&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
­	variables
Ўtrainable_variables
Џregularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses
Гkernel
	Дbias
!Е_jit_compiled_convolution_op"
_tf_keras_layer
А
/0
01
92
:3
;4
<5
I6
J7
S8
T9
U10
V11
c12
d13
m14
n15
o16
p17
}18
~19
20
21
22
23
24
25
Ё26
Ђ27
Ѓ28
Є29
Б30
В31
Л32
М33
Н34
О35
Ы36
Ь37
е38
ж39
з40
и41
х42
ц43
я44
№45
ё46
ђ47
џ48
49
50
51
52
53
54
55
Ѓ56
Є57
Ѕ58
І59
Г60
Д61"
trackable_list_wrapper

/0
01
92
:3
I4
J5
S6
T7
c8
d9
m10
n11
}12
~13
14
15
16
17
Ё18
Ђ19
Б20
В21
Л22
М23
Ы24
Ь25
е26
ж27
х28
ц29
я30
№31
џ32
33
34
35
36
37
Ѓ38
Є39
Г40
Д41"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
'_default_save_signature
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
н
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_32ъ
(__inference_decoder_layer_call_fn_137060
(__inference_decoder_layer_call_fn_138244
(__inference_decoder_layer_call_fn_138373
(__inference_decoder_layer_call_fn_137666П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЛtrace_0zМtrace_1zНtrace_2zОtrace_3
Щ
Пtrace_0
Рtrace_1
Сtrace_2
Тtrace_32ж
C__inference_decoder_layer_call_and_return_conditional_losses_138737
C__inference_decoder_layer_call_and_return_conditional_losses_139101
C__inference_decoder_layer_call_and_return_conditional_losses_137825
C__inference_decoder_layer_call_and_return_conditional_losses_137984П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zПtrace_0zРtrace_1zСtrace_2zТtrace_3
ЬBЩ
!__inference__wrapped_model_135581input_6"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
-
Уserving_default"
signature_map
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
В
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
њ
Щtrace_02л
4__inference_conv2d_transpose_22_layer_call_fn_139110Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЩtrace_0

Ъtrace_02і
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_139147Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЪtrace_0
5:3 2conv2d_transpose_22/kernel
&:$ 2conv2d_transpose_22/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
90
:1
;2
<3"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
у
аtrace_0
бtrace_12Ј
7__inference_batch_normalization_50_layer_call_fn_139160
7__inference_batch_normalization_50_layer_call_fn_139173Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zаtrace_0zбtrace_1

вtrace_0
гtrace_12о
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_139191
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_139209Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zвtrace_0zгtrace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_50/gamma
):' 2batch_normalization_50/beta
2:0  (2"batch_normalization_50/moving_mean
6:4  (2&batch_normalization_50/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
ѕ
йtrace_02ж
/__inference_leaky_re_lu_47_layer_call_fn_139214Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zйtrace_0

кtrace_02ё
J__inference_leaky_re_lu_47_layer_call_and_return_conditional_losses_139219Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zкtrace_0
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
њ
рtrace_02л
4__inference_conv2d_transpose_23_layer_call_fn_139228Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zрtrace_0

сtrace_02і
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_139261Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zсtrace_0
4:2@ 2conv2d_transpose_23/kernel
&:$@2conv2d_transpose_23/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
S0
T1
U2
V3"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
у
чtrace_0
шtrace_12Ј
7__inference_batch_normalization_51_layer_call_fn_139274
7__inference_batch_normalization_51_layer_call_fn_139287Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zчtrace_0zшtrace_1

щtrace_0
ъtrace_12о
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_139305
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_139323Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zщtrace_0zъtrace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_51/gamma
):'@2batch_normalization_51/beta
2:0@ (2"batch_normalization_51/moving_mean
6:4@ (2&batch_normalization_51/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
ѕ
№trace_02ж
/__inference_leaky_re_lu_48_layer_call_fn_139328Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z№trace_0

ёtrace_02ё
J__inference_leaky_re_lu_48_layer_call_and_return_conditional_losses_139333Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zёtrace_0
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
њ
їtrace_02л
4__inference_conv2d_transpose_24_layer_call_fn_139342Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zїtrace_0

јtrace_02і
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_139375Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zјtrace_0
5:3@2conv2d_transpose_24/kernel
':%2conv2d_transpose_24/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
<
m0
n1
o2
p3"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
у
ўtrace_0
џtrace_12Ј
7__inference_batch_normalization_52_layer_call_fn_139388
7__inference_batch_normalization_52_layer_call_fn_139401Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zўtrace_0zџtrace_1

trace_0
trace_12о
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_139419
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_139437Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
+:)2batch_normalization_52/gamma
*:(2batch_normalization_52/beta
3:1 (2"batch_normalization_52/moving_mean
7:5 (2&batch_normalization_52/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
ѕ
trace_02ж
/__inference_leaky_re_lu_49_layer_call_fn_139442Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ё
J__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_139447Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
њ
trace_02л
4__inference_conv2d_transpose_25_layer_call_fn_139456Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02і
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_139489Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
5:3@2conv2d_transpose_25/kernel
&:$@2conv2d_transpose_25/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
у
trace_0
trace_12Ј
7__inference_batch_normalization_53_layer_call_fn_139502
7__inference_batch_normalization_53_layer_call_fn_139515Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12о
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_139533
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_139551Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_53/gamma
):'@2batch_normalization_53/beta
2:0@ (2"batch_normalization_53/moving_mean
6:4@ (2&batch_normalization_53/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ѕ
trace_02ж
/__inference_leaky_re_lu_50_layer_call_fn_139556Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ё
J__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_139561Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
њ
Ѕtrace_02л
4__inference_conv2d_transpose_26_layer_call_fn_139570Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЅtrace_0

Іtrace_02і
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_139603Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zІtrace_0
4:2@@2conv2d_transpose_26/kernel
&:$@2conv2d_transpose_26/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
Ё0
Ђ1
Ѓ2
Є3"
trackable_list_wrapper
0
Ё0
Ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
у
Ќtrace_0
­trace_12Ј
7__inference_batch_normalization_54_layer_call_fn_139616
7__inference_batch_normalization_54_layer_call_fn_139629Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЌtrace_0z­trace_1

Ўtrace_0
Џtrace_12о
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_139647
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_139665Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЎtrace_0zЏtrace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_54/gamma
):'@2batch_normalization_54/beta
2:0@ (2"batch_normalization_54/moving_mean
6:4@ (2&batch_normalization_54/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
ѕ
Еtrace_02ж
/__inference_leaky_re_lu_51_layer_call_fn_139670Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0

Жtrace_02ё
J__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_139675Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЖtrace_0
0
Б0
В1"
trackable_list_wrapper
0
Б0
В1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
Ћ	variables
Ќtrainable_variables
­regularization_losses
Џ__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
њ
Мtrace_02л
4__inference_conv2d_transpose_27_layer_call_fn_139684Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zМtrace_0

Нtrace_02і
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_139717Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zНtrace_0
4:2 @2conv2d_transpose_27/kernel
&:$ 2conv2d_transpose_27/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
Л0
М1
Н2
О3"
trackable_list_wrapper
0
Л0
М1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
у
Уtrace_0
Фtrace_12Ј
7__inference_batch_normalization_55_layer_call_fn_139730
7__inference_batch_normalization_55_layer_call_fn_139743Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zУtrace_0zФtrace_1

Хtrace_0
Цtrace_12о
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_139761
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_139779Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zХtrace_0zЦtrace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_55/gamma
):' 2batch_normalization_55/beta
2:0  (2"batch_normalization_55/moving_mean
6:4  (2&batch_normalization_55/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
ѕ
Ьtrace_02ж
/__inference_leaky_re_lu_52_layer_call_fn_139784Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЬtrace_0

Эtrace_02ё
J__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_139789Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЭtrace_0
0
Ы0
Ь1"
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
њ
гtrace_02л
4__inference_conv2d_transpose_28_layer_call_fn_139798Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zгtrace_0

дtrace_02і
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_139831Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zдtrace_0
5:3 2conv2d_transpose_28/kernel
':%2conv2d_transpose_28/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
е0
ж1
з2
и3"
trackable_list_wrapper
0
е0
ж1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
Ю	variables
Яtrainable_variables
аregularization_losses
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
у
кtrace_0
лtrace_12Ј
7__inference_batch_normalization_56_layer_call_fn_139844
7__inference_batch_normalization_56_layer_call_fn_139857Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zкtrace_0zлtrace_1

мtrace_0
нtrace_12о
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_139875
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_139893Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zмtrace_0zнtrace_1
 "
trackable_list_wrapper
+:)2batch_normalization_56/gamma
*:(2batch_normalization_56/beta
3:1 (2"batch_normalization_56/moving_mean
7:5 (2&batch_normalization_56/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
оnon_trainable_variables
пlayers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
й	variables
кtrainable_variables
лregularization_losses
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
ѕ
уtrace_02ж
/__inference_leaky_re_lu_53_layer_call_fn_139898Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zуtrace_0

фtrace_02ё
J__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_139903Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zфtrace_0
0
х0
ц1"
trackable_list_wrapper
0
х0
ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
п	variables
рtrainable_variables
сregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
њ
ъtrace_02л
4__inference_conv2d_transpose_29_layer_call_fn_139912Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zъtrace_0

ыtrace_02і
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_139945Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zыtrace_0
5:3@2conv2d_transpose_29/kernel
&:$@2conv2d_transpose_29/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
я0
№1
ё2
ђ3"
trackable_list_wrapper
0
я0
№1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ьnon_trainable_variables
эlayers
юmetrics
 яlayer_regularization_losses
№layer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
у
ёtrace_0
ђtrace_12Ј
7__inference_batch_normalization_57_layer_call_fn_139958
7__inference_batch_normalization_57_layer_call_fn_139971Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zёtrace_0zђtrace_1

ѓtrace_0
єtrace_12о
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_139989
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_140007Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zѓtrace_0zєtrace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_57/gamma
):'@2batch_normalization_57/beta
2:0@ (2"batch_normalization_57/moving_mean
6:4@ (2&batch_normalization_57/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѕnon_trainable_variables
іlayers
їmetrics
 јlayer_regularization_losses
љlayer_metrics
ѓ	variables
єtrainable_variables
ѕregularization_losses
ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
ѕ
њtrace_02ж
/__inference_leaky_re_lu_54_layer_call_fn_140012Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zњtrace_0

ћtrace_02ё
J__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_140017Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zћtrace_0
0
џ0
1"
trackable_list_wrapper
0
џ0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ќnon_trainable_variables
§layers
ўmetrics
 џlayer_regularization_losses
layer_metrics
љ	variables
њtrainable_variables
ћregularization_losses
§__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
њ
trace_02л
4__inference_conv2d_transpose_30_layer_call_fn_140026Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02і
O__inference_conv2d_transpose_30_layer_call_and_return_conditional_losses_140059Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
4:2 @2conv2d_transpose_30/kernel
&:$ 2conv2d_transpose_30/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
у
trace_0
trace_12Ј
7__inference_batch_normalization_58_layer_call_fn_140072
7__inference_batch_normalization_58_layer_call_fn_140085Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12о
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_140103
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_140121Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_58/gamma
):' 2batch_normalization_58/beta
2:0  (2"batch_normalization_58/moving_mean
6:4  (2&batch_normalization_58/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ѕ
trace_02ж
/__inference_leaky_re_lu_55_layer_call_fn_140126Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ё
J__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_140131Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
њ
trace_02л
4__inference_conv2d_transpose_31_layer_call_fn_140140Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02і
O__inference_conv2d_transpose_31_layer_call_and_return_conditional_losses_140173Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
4:2  2conv2d_transpose_31/kernel
&:$ 2conv2d_transpose_31/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
@
Ѓ0
Є1
Ѕ2
І3"
trackable_list_wrapper
0
Ѓ0
Є1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
у
trace_0
 trace_12Ј
7__inference_batch_normalization_59_layer_call_fn_140186
7__inference_batch_normalization_59_layer_call_fn_140199Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0z trace_1

Ёtrace_0
Ђtrace_12о
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_140217
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_140235Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЁtrace_0zЂtrace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_59/gamma
):' 2batch_normalization_59/beta
2:0  (2"batch_normalization_59/moving_mean
6:4  (2&batch_normalization_59/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѓnon_trainable_variables
Єlayers
Ѕmetrics
 Іlayer_regularization_losses
Їlayer_metrics
Ї	variables
Јtrainable_variables
Љregularization_losses
Ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
ѕ
Јtrace_02ж
/__inference_leaky_re_lu_56_layer_call_fn_140240Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЈtrace_0

Љtrace_02ё
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_140245Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЉtrace_0
0
Г0
Д1"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
­	variables
Ўtrainable_variables
Џregularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
њ
Џtrace_02л
4__inference_conv2d_transpose_32_layer_call_fn_140254Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЏtrace_0

Аtrace_02і
O__inference_conv2d_transpose_32_layer_call_and_return_conditional_losses_140288Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zАtrace_0
4:2 2conv2d_transpose_32/kernel
&:$2conv2d_transpose_32/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
Ф
;0
<1
U2
V3
o4
p5
6
7
Ѓ8
Є9
Н10
О11
з12
и13
ё14
ђ15
16
17
Ѕ18
І19"
trackable_list_wrapper

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
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
њBї
(__inference_decoder_layer_call_fn_137060input_6"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
(__inference_decoder_layer_call_fn_138244inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
(__inference_decoder_layer_call_fn_138373inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
(__inference_decoder_layer_call_fn_137666input_6"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_decoder_layer_call_and_return_conditional_losses_138737inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_decoder_layer_call_and_return_conditional_losses_139101inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_decoder_layer_call_and_return_conditional_losses_137825input_6"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_decoder_layer_call_and_return_conditional_losses_137984input_6"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЫBШ
$__inference_signature_wrapper_138115input_6"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
шBх
4__inference_conv2d_transpose_22_layer_call_fn_139110inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_139147inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
7__inference_batch_normalization_50_layer_call_fn_139160inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
7__inference_batch_normalization_50_layer_call_fn_139173inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_139191inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_139209inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
уBр
/__inference_leaky_re_lu_47_layer_call_fn_139214inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_leaky_re_lu_47_layer_call_and_return_conditional_losses_139219inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
шBх
4__inference_conv2d_transpose_23_layer_call_fn_139228inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_139261inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
7__inference_batch_normalization_51_layer_call_fn_139274inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
7__inference_batch_normalization_51_layer_call_fn_139287inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_139305inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_139323inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
уBр
/__inference_leaky_re_lu_48_layer_call_fn_139328inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_leaky_re_lu_48_layer_call_and_return_conditional_losses_139333inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
шBх
4__inference_conv2d_transpose_24_layer_call_fn_139342inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_139375inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
7__inference_batch_normalization_52_layer_call_fn_139388inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
7__inference_batch_normalization_52_layer_call_fn_139401inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_139419inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_139437inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
уBр
/__inference_leaky_re_lu_49_layer_call_fn_139442inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_139447inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
шBх
4__inference_conv2d_transpose_25_layer_call_fn_139456inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_139489inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
7__inference_batch_normalization_53_layer_call_fn_139502inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
7__inference_batch_normalization_53_layer_call_fn_139515inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_139533inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_139551inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
уBр
/__inference_leaky_re_lu_50_layer_call_fn_139556inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_139561inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
шBх
4__inference_conv2d_transpose_26_layer_call_fn_139570inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_139603inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Ѓ0
Є1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
7__inference_batch_normalization_54_layer_call_fn_139616inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
7__inference_batch_normalization_54_layer_call_fn_139629inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_139647inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_139665inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
уBр
/__inference_leaky_re_lu_51_layer_call_fn_139670inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_139675inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
шBх
4__inference_conv2d_transpose_27_layer_call_fn_139684inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_139717inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Н0
О1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
7__inference_batch_normalization_55_layer_call_fn_139730inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
7__inference_batch_normalization_55_layer_call_fn_139743inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_139761inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_139779inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
уBр
/__inference_leaky_re_lu_52_layer_call_fn_139784inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_139789inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
шBх
4__inference_conv2d_transpose_28_layer_call_fn_139798inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_139831inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
з0
и1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
7__inference_batch_normalization_56_layer_call_fn_139844inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
7__inference_batch_normalization_56_layer_call_fn_139857inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_139875inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_139893inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
уBр
/__inference_leaky_re_lu_53_layer_call_fn_139898inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_139903inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
шBх
4__inference_conv2d_transpose_29_layer_call_fn_139912inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_139945inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
ё0
ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
7__inference_batch_normalization_57_layer_call_fn_139958inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
7__inference_batch_normalization_57_layer_call_fn_139971inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_139989inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_140007inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
уBр
/__inference_leaky_re_lu_54_layer_call_fn_140012inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_140017inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
шBх
4__inference_conv2d_transpose_30_layer_call_fn_140026inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_conv2d_transpose_30_layer_call_and_return_conditional_losses_140059inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
7__inference_batch_normalization_58_layer_call_fn_140072inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
7__inference_batch_normalization_58_layer_call_fn_140085inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_140103inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_140121inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
уBр
/__inference_leaky_re_lu_55_layer_call_fn_140126inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_140131inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
шBх
4__inference_conv2d_transpose_31_layer_call_fn_140140inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_conv2d_transpose_31_layer_call_and_return_conditional_losses_140173inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Ѕ0
І1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќBљ
7__inference_batch_normalization_59_layer_call_fn_140186inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
7__inference_batch_normalization_59_layer_call_fn_140199inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_140217inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_140235inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
уBр
/__inference_leaky_re_lu_56_layer_call_fn_140240inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_140245inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
шBх
4__inference_conv2d_transpose_32_layer_call_fn_140254inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_conv2d_transpose_32_layer_call_and_return_conditional_losses_140288inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
  
!__inference__wrapped_model_135581њh/09:;<IJSTUVcdmnop}~ЁЂЃЄБВЛМНОЫЬежзихця№ёђџЃЄЅІГД9Ђ6
/Ђ,
*'
input_6џџџџџџџџџ
Њ "SЊP
N
conv2d_transpose_3274
conv2d_transpose_32џџџџџџџџџэ
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1391919:;<MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 э
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1392099:;<MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Х
7__inference_batch_normalization_50_layer_call_fn_1391609:;<MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Х
7__inference_batch_normalization_50_layer_call_fn_1391739:;<MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ э
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_139305STUVMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 э
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_139323STUVMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Х
7__inference_batch_normalization_51_layer_call_fn_139274STUVMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Х
7__inference_batch_normalization_51_layer_call_fn_139287STUVMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@я
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_139419mnopNЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 я
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_139437mnopNЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
7__inference_batch_normalization_52_layer_call_fn_139388mnopNЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџЧ
7__inference_batch_normalization_52_layer_call_fn_139401mnopNЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџё
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_139533MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 ё
R__inference_batch_normalization_53_layer_call_and_return_conditional_losses_139551MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Щ
7__inference_batch_normalization_53_layer_call_fn_139502MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Щ
7__inference_batch_normalization_53_layer_call_fn_139515MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@ё
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_139647ЁЂЃЄMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 ё
R__inference_batch_normalization_54_layer_call_and_return_conditional_losses_139665ЁЂЃЄMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Щ
7__inference_batch_normalization_54_layer_call_fn_139616ЁЂЃЄMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Щ
7__inference_batch_normalization_54_layer_call_fn_139629ЁЂЃЄMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@ё
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_139761ЛМНОMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 ё
R__inference_batch_normalization_55_layer_call_and_return_conditional_losses_139779ЛМНОMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Щ
7__inference_batch_normalization_55_layer_call_fn_139730ЛМНОMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Щ
7__inference_batch_normalization_55_layer_call_fn_139743ЛМНОMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ѓ
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_139875ежзиNЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ѓ
R__inference_batch_normalization_56_layer_call_and_return_conditional_losses_139893ежзиNЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ы
7__inference_batch_normalization_56_layer_call_fn_139844ежзиNЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџЫ
7__inference_batch_normalization_56_layer_call_fn_139857ежзиNЂK
DЂA
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџё
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_139989я№ёђMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 ё
R__inference_batch_normalization_57_layer_call_and_return_conditional_losses_140007я№ёђMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Щ
7__inference_batch_normalization_57_layer_call_fn_139958я№ёђMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Щ
7__inference_batch_normalization_57_layer_call_fn_139971я№ёђMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@ё
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_140103MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 ё
R__inference_batch_normalization_58_layer_call_and_return_conditional_losses_140121MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Щ
7__inference_batch_normalization_58_layer_call_fn_140072MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Щ
7__inference_batch_normalization_58_layer_call_fn_140085MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ё
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_140217ЃЄЅІMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 ё
R__inference_batch_normalization_59_layer_call_and_return_conditional_losses_140235ЃЄЅІMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Щ
7__inference_batch_normalization_59_layer_call_fn_140186ЃЄЅІMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Щ
7__inference_batch_normalization_59_layer_call_fn_140199ЃЄЅІMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ х
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_139147/0JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Н
4__inference_conv2d_transpose_22_layer_call_fn_139110/0JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ф
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_139261IJIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 М
4__inference_conv2d_transpose_23_layer_call_fn_139228IJIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@х
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_139375cdIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Н
4__inference_conv2d_transpose_24_layer_call_fn_139342cdIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџх
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_139489}~JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Н
4__inference_conv2d_transpose_25_layer_call_fn_139456}~JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@ц
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_139603IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 О
4__inference_conv2d_transpose_26_layer_call_fn_139570IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@ц
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_139717БВIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 О
4__inference_conv2d_transpose_27_layer_call_fn_139684БВIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ч
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_139831ЫЬIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 П
4__inference_conv2d_transpose_28_layer_call_fn_139798ЫЬIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџч
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_139945хцJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 П
4__inference_conv2d_transpose_29_layer_call_fn_139912хцJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@ц
O__inference_conv2d_transpose_30_layer_call_and_return_conditional_losses_140059џIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 О
4__inference_conv2d_transpose_30_layer_call_fn_140026џIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ц
O__inference_conv2d_transpose_31_layer_call_and_return_conditional_losses_140173IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 О
4__inference_conv2d_transpose_31_layer_call_fn_140140IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ц
O__inference_conv2d_transpose_32_layer_call_and_return_conditional_losses_140288ГДIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 О
4__inference_conv2d_transpose_32_layer_call_fn_140254ГДIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџІ
C__inference_decoder_layer_call_and_return_conditional_losses_137825оh/09:;<IJSTUVcdmnop}~ЁЂЃЄБВЛМНОЫЬежзихця№ёђџЃЄЅІГДAЂ>
7Ђ4
*'
input_6џџџџџџџџџ
p 

 
Њ "/Ђ,
%"
0џџџџџџџџџ
 І
C__inference_decoder_layer_call_and_return_conditional_losses_137984оh/09:;<IJSTUVcdmnop}~ЁЂЃЄБВЛМНОЫЬежзихця№ёђџЃЄЅІГДAЂ>
7Ђ4
*'
input_6џџџџџџџџџ
p

 
Њ "/Ђ,
%"
0џџџџџџџџџ
 Ѕ
C__inference_decoder_layer_call_and_return_conditional_losses_138737нh/09:;<IJSTUVcdmnop}~ЁЂЃЄБВЛМНОЫЬежзихця№ёђџЃЄЅІГД@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ
p 

 
Њ "/Ђ,
%"
0џџџџџџџџџ
 Ѕ
C__inference_decoder_layer_call_and_return_conditional_losses_139101нh/09:;<IJSTUVcdmnop}~ЁЂЃЄБВЛМНОЫЬежзихця№ёђџЃЄЅІГД@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ
p

 
Њ "/Ђ,
%"
0џџџџџџџџџ
 ў
(__inference_decoder_layer_call_fn_137060бh/09:;<IJSTUVcdmnop}~ЁЂЃЄБВЛМНОЫЬежзихця№ёђџЃЄЅІГДAЂ>
7Ђ4
*'
input_6џџџџџџџџџ
p 

 
Њ ""џџџџџџџџџў
(__inference_decoder_layer_call_fn_137666бh/09:;<IJSTUVcdmnop}~ЁЂЃЄБВЛМНОЫЬежзихця№ёђџЃЄЅІГДAЂ>
7Ђ4
*'
input_6џџџџџџџџџ
p

 
Њ ""џџџџџџџџџ§
(__inference_decoder_layer_call_fn_138244аh/09:;<IJSTUVcdmnop}~ЁЂЃЄБВЛМНОЫЬежзихця№ёђџЃЄЅІГД@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ
p 

 
Њ ""џџџџџџџџџ§
(__inference_decoder_layer_call_fn_138373аh/09:;<IJSTUVcdmnop}~ЁЂЃЄБВЛМНОЫЬежзихця№ёђџЃЄЅІГД@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ
p

 
Њ ""џџџџџџџџџЖ
J__inference_leaky_re_lu_47_layer_call_and_return_conditional_losses_139219h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
/__inference_leaky_re_lu_47_layer_call_fn_139214[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ Ж
J__inference_leaky_re_lu_48_layer_call_and_return_conditional_losses_139333h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
/__inference_leaky_re_lu_48_layer_call_fn_139328[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@И
J__inference_leaky_re_lu_49_layer_call_and_return_conditional_losses_139447j8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
/__inference_leaky_re_lu_49_layer_call_fn_139442]8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЖ
J__inference_leaky_re_lu_50_layer_call_and_return_conditional_losses_139561h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
/__inference_leaky_re_lu_50_layer_call_fn_139556[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@Ж
J__inference_leaky_re_lu_51_layer_call_and_return_conditional_losses_139675h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
/__inference_leaky_re_lu_51_layer_call_fn_139670[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@Ж
J__inference_leaky_re_lu_52_layer_call_and_return_conditional_losses_139789h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
/__inference_leaky_re_lu_52_layer_call_fn_139784[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ И
J__inference_leaky_re_lu_53_layer_call_and_return_conditional_losses_139903j8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ  
Њ ".Ђ+
$!
0џџџџџџџџџ  
 
/__inference_leaky_re_lu_53_layer_call_fn_139898]8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ  
Њ "!џџџџџџџџџ  Ж
J__inference_leaky_re_lu_54_layer_call_and_return_conditional_losses_140017h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  @
Њ "-Ђ*
# 
0џџџџџџџџџ  @
 
/__inference_leaky_re_lu_54_layer_call_fn_140012[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ  @
Њ " џџџџџџџџџ  @Ж
J__inference_leaky_re_lu_55_layer_call_and_return_conditional_losses_140131h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@ 
Њ "-Ђ*
# 
0џџџџџџџџџ@@ 
 
/__inference_leaky_re_lu_55_layer_call_fn_140126[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@@ 
Њ " џџџџџџџџџ@@ К
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_140245l9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ 
Њ "/Ђ,
%"
0џџџџџџџџџ 
 
/__inference_leaky_re_lu_56_layer_call_fn_140240_9Ђ6
/Ђ,
*'
inputsџџџџџџџџџ 
Њ ""џџџџџџџџџ Ў
$__inference_signature_wrapper_138115h/09:;<IJSTUVcdmnop}~ЁЂЃЄБВЛМНОЫЬежзихця№ёђџЃЄЅІГДDЂA
Ђ 
:Њ7
5
input_6*'
input_6џџџџџџџџџ"SЊP
N
conv2d_transpose_3274
conv2d_transpose_32џџџџџџџџџ