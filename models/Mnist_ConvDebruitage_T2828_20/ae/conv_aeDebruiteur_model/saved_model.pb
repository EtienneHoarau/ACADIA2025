Кш
ђс
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
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
Џ
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
└
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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
ѓ
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Ї
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
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
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.10.02unknown8┘╦
ќ
Nadam/conv2d_transpose_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/conv2d_transpose_5/bias/v
Ј
3Nadam/conv2d_transpose_5/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose_5/bias/v*
_output_shapes
:*
dtype0
д
!Nadam/conv2d_transpose_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Nadam/conv2d_transpose_5/kernel/v
Ъ
5Nadam/conv2d_transpose_5/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/conv2d_transpose_5/kernel/v*&
_output_shapes
:*
dtype0
ќ
Nadam/conv2d_transpose_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/conv2d_transpose_4/bias/v
Ј
3Nadam/conv2d_transpose_4/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose_4/bias/v*
_output_shapes
:*
dtype0
д
!Nadam/conv2d_transpose_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Nadam/conv2d_transpose_4/kernel/v
Ъ
5Nadam/conv2d_transpose_4/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/conv2d_transpose_4/kernel/v*&
_output_shapes
:*
dtype0
ќ
Nadam/conv2d_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/conv2d_transpose_3/bias/v
Ј
3Nadam/conv2d_transpose_3/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose_3/bias/v*
_output_shapes
:*
dtype0
д
!Nadam/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Nadam/conv2d_transpose_3/kernel/v
Ъ
5Nadam/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/conv2d_transpose_3/kernel/v*&
_output_shapes
:*
dtype0
Ђ
Nadam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ*%
shared_nameNadam/dense_1/bias/v
z
(Nadam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_1/bias/v*
_output_shapes	
:љ*
dtype0
Ѕ
Nadam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љ*'
shared_nameNadam/dense_1/kernel/v
ѓ
*Nadam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_1/kernel/v*
_output_shapes
:	љ*
dtype0
ѓ
Nadam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/conv2d_7/bias/v
{
)Nadam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_7/bias/v*
_output_shapes
:*
dtype0
њ
Nadam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameNadam/conv2d_7/kernel/v
І
+Nadam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_7/kernel/v*&
_output_shapes
:@*
dtype0
ѓ
Nadam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameNadam/conv2d_6/bias/v
{
)Nadam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_6/bias/v*
_output_shapes
:@*
dtype0
њ
Nadam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameNadam/conv2d_6/kernel/v
І
+Nadam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_6/kernel/v*&
_output_shapes
: @*
dtype0
ѓ
Nadam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d_5/bias/v
{
)Nadam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_5/bias/v*
_output_shapes
: *
dtype0
њ
Nadam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameNadam/conv2d_5/kernel/v
І
+Nadam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_5/kernel/v*&
_output_shapes
: *
dtype0
ѓ
Nadam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/conv2d_4/bias/v
{
)Nadam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_4/bias/v*
_output_shapes
:*
dtype0
њ
Nadam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameNadam/conv2d_4/kernel/v
І
+Nadam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_4/kernel/v*&
_output_shapes
:*
dtype0
ќ
Nadam/conv2d_transpose_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/conv2d_transpose_5/bias/m
Ј
3Nadam/conv2d_transpose_5/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose_5/bias/m*
_output_shapes
:*
dtype0
д
!Nadam/conv2d_transpose_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Nadam/conv2d_transpose_5/kernel/m
Ъ
5Nadam/conv2d_transpose_5/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/conv2d_transpose_5/kernel/m*&
_output_shapes
:*
dtype0
ќ
Nadam/conv2d_transpose_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/conv2d_transpose_4/bias/m
Ј
3Nadam/conv2d_transpose_4/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose_4/bias/m*
_output_shapes
:*
dtype0
д
!Nadam/conv2d_transpose_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Nadam/conv2d_transpose_4/kernel/m
Ъ
5Nadam/conv2d_transpose_4/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/conv2d_transpose_4/kernel/m*&
_output_shapes
:*
dtype0
ќ
Nadam/conv2d_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/conv2d_transpose_3/bias/m
Ј
3Nadam/conv2d_transpose_3/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose_3/bias/m*
_output_shapes
:*
dtype0
д
!Nadam/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Nadam/conv2d_transpose_3/kernel/m
Ъ
5Nadam/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/conv2d_transpose_3/kernel/m*&
_output_shapes
:*
dtype0
Ђ
Nadam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ*%
shared_nameNadam/dense_1/bias/m
z
(Nadam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_1/bias/m*
_output_shapes	
:љ*
dtype0
Ѕ
Nadam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љ*'
shared_nameNadam/dense_1/kernel/m
ѓ
*Nadam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_1/kernel/m*
_output_shapes
:	љ*
dtype0
ѓ
Nadam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/conv2d_7/bias/m
{
)Nadam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_7/bias/m*
_output_shapes
:*
dtype0
њ
Nadam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameNadam/conv2d_7/kernel/m
І
+Nadam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_7/kernel/m*&
_output_shapes
:@*
dtype0
ѓ
Nadam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameNadam/conv2d_6/bias/m
{
)Nadam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_6/bias/m*
_output_shapes
:@*
dtype0
њ
Nadam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameNadam/conv2d_6/kernel/m
І
+Nadam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_6/kernel/m*&
_output_shapes
: @*
dtype0
ѓ
Nadam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d_5/bias/m
{
)Nadam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_5/bias/m*
_output_shapes
: *
dtype0
њ
Nadam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameNadam/conv2d_5/kernel/m
І
+Nadam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_5/kernel/m*&
_output_shapes
: *
dtype0
ѓ
Nadam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/conv2d_4/bias/m
{
)Nadam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_4/bias/m*
_output_shapes
:*
dtype0
њ
Nadam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameNadam/conv2d_4/kernel/m
І
+Nadam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_4/kernel/m*&
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
є
conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_5/bias

+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes
:*
dtype0
ќ
conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_5/kernel
Ј
-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*&
_output_shapes
:*
dtype0
є
conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_4/bias

+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes
:*
dtype0
ќ
conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_4/kernel
Ј
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
:*
dtype0
є
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:*
dtype0
ќ
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_3/kernel
Ј
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
:*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:љ*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љ*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	љ*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:*
dtype0
ѓ
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
ѓ
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0
ѓ
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
: *
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0
ѓ
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
і
serving_default_input_3Placeholder*/
_output_shapes
:         *
dtype0*$
shape:         
ћ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_1/kerneldense_1/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_transpose_5/kernelconv2d_transpose_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_365700

NoOpNoOp
ює
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*оЁ
value╦ЁBКЁ B┐Ё
Д
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
* 
ь
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
к
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
 layer_with_weights-3
 layer-5
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
z
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615*
z
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615*
* 
░
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
<trace_0
=trace_1
>trace_2
?trace_3* 
6
@trace_0
Atrace_1
Btrace_2
Ctrace_3* 
* 
ў
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_rate
Imomentum_cache'mа(mА)mб*mБ+mц,mЦ-mд.mД/mе0mЕ1mф2mФ3mг4mГ5m«6m»'v░(v▒)v▓*v│+v┤,vх-vХ.vи/vИ0v╣1v║2v╗3v╝4vй5vЙ6v┐*

Jserving_default* 
╚
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

'kernel
(bias
 Q_jit_compiled_convolution_op*
ј
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses* 
╚
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

)kernel
*bias
 ^_jit_compiled_convolution_op*
ј
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses* 
╚
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

+kernel
,bias
 k_jit_compiled_convolution_op*
ј
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 
╚
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

-kernel
.bias
 x_jit_compiled_convolution_op*
ј
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses* 
<
'0
(1
)2
*3
+4
,5
-6
.7*
<
'0
(1
)2
*3
+4
,5
-6
.7*
* 
Ќ
non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
ёtrace_0
Ёtrace_1
єtrace_2
Єtrace_3* 
:
ѕtrace_0
Ѕtrace_1
іtrace_2
Іtrace_3* 
* 
г
ї	variables
Їtrainable_variables
јregularization_losses
Ј	keras_api
љ__call__
+Љ&call_and_return_all_conditional_losses

/kernel
0bias*
ћ
њ	variables
Њtrainable_variables
ћregularization_losses
Ћ	keras_api
ќ__call__
+Ќ&call_and_return_all_conditional_losses* 
¤
ў	variables
Ўtrainable_variables
џregularization_losses
Џ	keras_api
ю__call__
+Ю&call_and_return_all_conditional_losses

1kernel
2bias
!ъ_jit_compiled_convolution_op*
¤
Ъ	variables
аtrainable_variables
Аregularization_losses
б	keras_api
Б__call__
+ц&call_and_return_all_conditional_losses

3kernel
4bias
!Ц_jit_compiled_convolution_op*
¤
д	variables
Дtrainable_variables
еregularization_losses
Е	keras_api
ф__call__
+Ф&call_and_return_all_conditional_losses

5kernel
6bias
!г_jit_compiled_convolution_op*
<
/0
01
12
23
34
45
56
67*
<
/0
01
12
23
34
45
56
67*
* 
ў
Гnon_trainable_variables
«layers
»metrics
 ░layer_regularization_losses
▒layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
:
▓trace_0
│trace_1
┤trace_2
хtrace_3* 
:
Хtrace_0
иtrace_1
Иtrace_2
╣trace_3* 
OI
VARIABLE_VALUEconv2d_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_4/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_5/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_5/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_6/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_6/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_7/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_7/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_3/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_3/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_4/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_4/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_5/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_5/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

║0
╗1*
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
MG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE*
* 

'0
(1*

'0
(1*
* 
ў
╝non_trainable_variables
йlayers
Йmetrics
 ┐layer_regularization_losses
└layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

┴trace_0* 

┬trace_0* 
* 
* 
* 
* 
ќ
├non_trainable_variables
─layers
┼metrics
 кlayer_regularization_losses
Кlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses* 

╚trace_0* 

╔trace_0* 

)0
*1*

)0
*1*
* 
ў
╩non_trainable_variables
╦layers
╠metrics
 ═layer_regularization_losses
╬layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

¤trace_0* 

лtrace_0* 
* 
* 
* 
* 
ќ
Лnon_trainable_variables
мlayers
Мmetrics
 нlayer_regularization_losses
Нlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

оtrace_0* 

Оtrace_0* 

+0
,1*

+0
,1*
* 
ў
пnon_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

Пtrace_0* 

яtrace_0* 
* 
* 
* 
* 
ќ
▀non_trainable_variables
Яlayers
рmetrics
 Рlayer_regularization_losses
сlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 

Сtrace_0* 

тtrace_0* 

-0
.1*

-0
.1*
* 
ў
Тnon_trainable_variables
уlayers
Уmetrics
 жlayer_regularization_losses
Жlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

вtrace_0* 

Вtrace_0* 
* 
* 
* 
* 
ќ
ьnon_trainable_variables
Ьlayers
№metrics
 ­layer_regularization_losses
ыlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses* 

Ыtrace_0* 

зtrace_0* 
* 
C
0
1
2
3
4
5
6
7
8*
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
/0
01*

/0
01*
* 
ъ
Зnon_trainable_variables
шlayers
Шmetrics
 эlayer_regularization_losses
Эlayer_metrics
ї	variables
Їtrainable_variables
јregularization_losses
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses*

щtrace_0* 

Щtrace_0* 
* 
* 
* 
ю
чnon_trainable_variables
Чlayers
§metrics
 ■layer_regularization_losses
 layer_metrics
њ	variables
Њtrainable_variables
ћregularization_losses
ќ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses* 

ђtrace_0* 

Ђtrace_0* 

10
21*

10
21*
* 
ъ
ѓnon_trainable_variables
Ѓlayers
ёmetrics
 Ёlayer_regularization_losses
єlayer_metrics
ў	variables
Ўtrainable_variables
џregularization_losses
ю__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses*

Єtrace_0* 

ѕtrace_0* 
* 

30
41*

30
41*
* 
ъ
Ѕnon_trainable_variables
іlayers
Іmetrics
 їlayer_regularization_losses
Їlayer_metrics
Ъ	variables
аtrainable_variables
Аregularization_losses
Б__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses*

јtrace_0* 

Јtrace_0* 
* 

50
61*

50
61*
* 
ъ
љnon_trainable_variables
Љlayers
њmetrics
 Њlayer_regularization_losses
ћlayer_metrics
д	variables
Дtrainable_variables
еregularization_losses
ф__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses*

Ћtrace_0* 

ќtrace_0* 
* 
* 
.
0
1
2
3
4
 5*
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
<
Ќ	variables
ў	keras_api

Ўtotal

џcount*
M
Џ	variables
ю	keras_api

Юtotal

ъcount
Ъ
_fn_kwargs*
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
Ў0
џ1*

Ќ	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ю0
ъ1*

Џ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
sm
VARIABLE_VALUENadam/conv2d_4/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_4/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_5/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_5/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_6/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_6/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_7/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_7/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUENadam/dense_1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUENadam/dense_1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Nadam/conv2d_transpose_3/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/conv2d_transpose_3/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Nadam/conv2d_transpose_4/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/conv2d_transpose_4/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Nadam/conv2d_transpose_5/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/conv2d_transpose_5/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_4/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_4/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_5/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_5/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_6/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_6/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_7/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_7/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUENadam/dense_1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUENadam/dense_1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Nadam/conv2d_transpose_3/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/conv2d_transpose_3/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Nadam/conv2d_transpose_4/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/conv2d_transpose_4/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Nadam/conv2d_transpose_5/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/conv2d_transpose_5/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
І
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOp-conv2d_transpose_5/kernel/Read/ReadVariableOp+conv2d_transpose_5/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Nadam/conv2d_4/kernel/m/Read/ReadVariableOp)Nadam/conv2d_4/bias/m/Read/ReadVariableOp+Nadam/conv2d_5/kernel/m/Read/ReadVariableOp)Nadam/conv2d_5/bias/m/Read/ReadVariableOp+Nadam/conv2d_6/kernel/m/Read/ReadVariableOp)Nadam/conv2d_6/bias/m/Read/ReadVariableOp+Nadam/conv2d_7/kernel/m/Read/ReadVariableOp)Nadam/conv2d_7/bias/m/Read/ReadVariableOp*Nadam/dense_1/kernel/m/Read/ReadVariableOp(Nadam/dense_1/bias/m/Read/ReadVariableOp5Nadam/conv2d_transpose_3/kernel/m/Read/ReadVariableOp3Nadam/conv2d_transpose_3/bias/m/Read/ReadVariableOp5Nadam/conv2d_transpose_4/kernel/m/Read/ReadVariableOp3Nadam/conv2d_transpose_4/bias/m/Read/ReadVariableOp5Nadam/conv2d_transpose_5/kernel/m/Read/ReadVariableOp3Nadam/conv2d_transpose_5/bias/m/Read/ReadVariableOp+Nadam/conv2d_4/kernel/v/Read/ReadVariableOp)Nadam/conv2d_4/bias/v/Read/ReadVariableOp+Nadam/conv2d_5/kernel/v/Read/ReadVariableOp)Nadam/conv2d_5/bias/v/Read/ReadVariableOp+Nadam/conv2d_6/kernel/v/Read/ReadVariableOp)Nadam/conv2d_6/bias/v/Read/ReadVariableOp+Nadam/conv2d_7/kernel/v/Read/ReadVariableOp)Nadam/conv2d_7/bias/v/Read/ReadVariableOp*Nadam/dense_1/kernel/v/Read/ReadVariableOp(Nadam/dense_1/bias/v/Read/ReadVariableOp5Nadam/conv2d_transpose_3/kernel/v/Read/ReadVariableOp3Nadam/conv2d_transpose_3/bias/v/Read/ReadVariableOp5Nadam/conv2d_transpose_4/kernel/v/Read/ReadVariableOp3Nadam/conv2d_transpose_4/bias/v/Read/ReadVariableOp5Nadam/conv2d_transpose_5/kernel/v/Read/ReadVariableOp3Nadam/conv2d_transpose_5/bias/v/Read/ReadVariableOpConst*G
Tin@
>2<	*
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
GPU 2J 8ѓ *(
f#R!
__inference__traced_save_366802
■
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_1/kerneldense_1/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_transpose_5/kernelconv2d_transpose_5/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotal_1count_1totalcountNadam/conv2d_4/kernel/mNadam/conv2d_4/bias/mNadam/conv2d_5/kernel/mNadam/conv2d_5/bias/mNadam/conv2d_6/kernel/mNadam/conv2d_6/bias/mNadam/conv2d_7/kernel/mNadam/conv2d_7/bias/mNadam/dense_1/kernel/mNadam/dense_1/bias/m!Nadam/conv2d_transpose_3/kernel/mNadam/conv2d_transpose_3/bias/m!Nadam/conv2d_transpose_4/kernel/mNadam/conv2d_transpose_4/bias/m!Nadam/conv2d_transpose_5/kernel/mNadam/conv2d_transpose_5/bias/mNadam/conv2d_4/kernel/vNadam/conv2d_4/bias/vNadam/conv2d_5/kernel/vNadam/conv2d_5/bias/vNadam/conv2d_6/kernel/vNadam/conv2d_6/bias/vNadam/conv2d_7/kernel/vNadam/conv2d_7/bias/vNadam/dense_1/kernel/vNadam/dense_1/bias/v!Nadam/conv2d_transpose_3/kernel/vNadam/conv2d_transpose_3/bias/v!Nadam/conv2d_transpose_4/kernel/vNadam/conv2d_transpose_4/bias/v!Nadam/conv2d_transpose_5/kernel/vNadam/conv2d_transpose_5/bias/v*F
Tin?
=2;*
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
GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_366986зб
─!
Џ
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_365074

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
value	B :y
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ѓ
§
D__inference_conv2d_6_layer_call_and_return_conditional_losses_364755

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_366374

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ь-
┌
C__inference_encoder_layer_call_and_return_conditional_losses_366114

inputsA
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:A
'conv2d_5_conv2d_readvariableop_resource: 6
(conv2d_5_biasadd_readvariableop_resource: A
'conv2d_6_conv2d_readvariableop_resource: @6
(conv2d_6_biasadd_readvariableop_resource:@A
'conv2d_7_conv2d_readvariableop_resource:@6
(conv2d_7_biasadd_readvariableop_resource:
identityѕбconv2d_4/BiasAdd/ReadVariableOpбconv2d_4/Conv2D/ReadVariableOpбconv2d_5/BiasAdd/ReadVariableOpбconv2d_5/Conv2D/ReadVariableOpбconv2d_6/BiasAdd/ReadVariableOpбconv2d_6/Conv2D/ReadVariableOpбconv2d_7/BiasAdd/ReadVariableOpбconv2d_7/Conv2D/ReadVariableOpј
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ё
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         j
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         г
max_pooling2d_3/MaxPoolMaxPoolconv2d_4/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
ј
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0┼
conv2d_5/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ё
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          j
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:          г
max_pooling2d_4/MaxPoolMaxPoolconv2d_5/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
ј
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0┼
conv2d_6/Conv2DConv2D max_pooling2d_4/MaxPool:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ё
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ў
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         @г
max_pooling2d_5/MaxPoolMaxPoolconv2d_6/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ј
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0┼
conv2d_7/Conv2DConv2D max_pooling2d_5/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ё
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         ѓ
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ▓
global_average_pooling2d_1/MeanMeanconv2d_7/Relu:activations:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         w
IdentityIdentity(global_average_pooling2d_1/Mean:output:0^NoOp*
T0*'
_output_shapes
:         м
NoOpNoOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ж	
м
(__inference_decoder_layer_call_fn_365303
input_4
unknown:	љ
	unknown_0:	љ#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identityѕбStatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_365263w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_4
Ѓ
§
D__inference_conv2d_6_layer_call_and_return_conditional_losses_366394

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
їА
Ч
C__inference_model_1_layer_call_and_return_conditional_losses_365886

inputsI
/encoder_conv2d_4_conv2d_readvariableop_resource:>
0encoder_conv2d_4_biasadd_readvariableop_resource:I
/encoder_conv2d_5_conv2d_readvariableop_resource: >
0encoder_conv2d_5_biasadd_readvariableop_resource: I
/encoder_conv2d_6_conv2d_readvariableop_resource: @>
0encoder_conv2d_6_biasadd_readvariableop_resource:@I
/encoder_conv2d_7_conv2d_readvariableop_resource:@>
0encoder_conv2d_7_biasadd_readvariableop_resource:A
.decoder_dense_1_matmul_readvariableop_resource:	љ>
/decoder_dense_1_biasadd_readvariableop_resource:	љ]
Cdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_3_biasadd_readvariableop_resource:]
Cdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_4_biasadd_readvariableop_resource:]
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_5_biasadd_readvariableop_resource:
identityѕб1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpб:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpб1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpб:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpб1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpб:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpб&decoder/dense_1/BiasAdd/ReadVariableOpб%decoder/dense_1/MatMul/ReadVariableOpб'encoder/conv2d_4/BiasAdd/ReadVariableOpб&encoder/conv2d_4/Conv2D/ReadVariableOpб'encoder/conv2d_5/BiasAdd/ReadVariableOpб&encoder/conv2d_5/Conv2D/ReadVariableOpб'encoder/conv2d_6/BiasAdd/ReadVariableOpб&encoder/conv2d_6/Conv2D/ReadVariableOpб'encoder/conv2d_7/BiasAdd/ReadVariableOpб&encoder/conv2d_7/Conv2D/ReadVariableOpъ
&encoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╗
encoder/conv2d_4/Conv2DConv2Dinputs.encoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ћ
'encoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0░
encoder/conv2d_4/BiasAddBiasAdd encoder/conv2d_4/Conv2D:output:0/encoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
encoder/conv2d_4/ReluRelu!encoder/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         ╝
encoder/max_pooling2d_3/MaxPoolMaxPool#encoder/conv2d_4/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
ъ
&encoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0П
encoder/conv2d_5/Conv2DConv2D(encoder/max_pooling2d_3/MaxPool:output:0.encoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ћ
'encoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0░
encoder/conv2d_5/BiasAddBiasAdd encoder/conv2d_5/Conv2D:output:0/encoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          z
encoder/conv2d_5/ReluRelu!encoder/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
encoder/max_pooling2d_4/MaxPoolMaxPool#encoder/conv2d_5/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
ъ
&encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0П
encoder/conv2d_6/Conv2DConv2D(encoder/max_pooling2d_4/MaxPool:output:0.encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ћ
'encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0░
encoder/conv2d_6/BiasAddBiasAdd encoder/conv2d_6/Conv2D:output:0/encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @z
encoder/conv2d_6/ReluRelu!encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         @╝
encoder/max_pooling2d_5/MaxPoolMaxPool#encoder/conv2d_6/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ъ
&encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0П
encoder/conv2d_7/Conv2DConv2D(encoder/max_pooling2d_5/MaxPool:output:0.encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ћ
'encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0░
encoder/conv2d_7/BiasAddBiasAdd encoder/conv2d_7/Conv2D:output:0/encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
encoder/conv2d_7/ReluRelu!encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         і
9encoder/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ╩
'encoder/global_average_pooling2d_1/MeanMean#encoder/conv2d_7/Relu:activations:0Bencoder/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Ћ
%decoder/dense_1/MatMul/ReadVariableOpReadVariableOp.decoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0┤
decoder/dense_1/MatMulMatMul0encoder/global_average_pooling2d_1/Mean:output:0-decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЊ
&decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Д
decoder/dense_1/BiasAddBiasAdd decoder/dense_1/MatMul:product:0.decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љg
decoder/reshape_1/ShapeShape decoder/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:o
%decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
decoder/reshape_1/strided_sliceStridedSlice decoder/reshape_1/Shape:output:0.decoder/reshape_1/strided_slice/stack:output:00decoder/reshape_1/strided_slice/stack_1:output:00decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ѓ
decoder/reshape_1/Reshape/shapePack(decoder/reshape_1/strided_slice:output:0*decoder/reshape_1/Reshape/shape/1:output:0*decoder/reshape_1/Reshape/shape/2:output:0*decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ф
decoder/reshape_1/ReshapeReshape decoder/dense_1/BiasAdd:output:0(decoder/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         r
 decoder/conv2d_transpose_3/ShapeShape"decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
(decoder/conv2d_transpose_3/strided_sliceStridedSlice)decoder/conv2d_transpose_3/Shape:output:07decoder/conv2d_transpose_3/strided_slice/stack:output:09decoder/conv2d_transpose_3/strided_slice/stack_1:output:09decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :љ
 decoder/conv2d_transpose_3/stackPack1decoder/conv2d_transpose_3/strided_slice:output:0+decoder/conv2d_transpose_3/stack/1:output:0+decoder/conv2d_transpose_3/stack/2:output:0+decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice)decoder/conv2d_transpose_3/stack:output:09decoder/conv2d_transpose_3/strided_slice_1/stack:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskк
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0И
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_3/stack:output:0Bdecoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0"decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
е
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
"decoder/conv2d_transpose_3/BiasAddBiasAdd4decoder/conv2d_transpose_3/conv2d_transpose:output:09decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ј
decoder/conv2d_transpose_3/ReluRelu+decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:         }
 decoder/conv2d_transpose_4/ShapeShape-decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
(decoder/conv2d_transpose_4/strided_sliceStridedSlice)decoder/conv2d_transpose_4/Shape:output:07decoder/conv2d_transpose_4/strided_slice/stack:output:09decoder/conv2d_transpose_4/strided_slice/stack_1:output:09decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :љ
 decoder/conv2d_transpose_4/stackPack1decoder/conv2d_transpose_4/strided_slice:output:0+decoder/conv2d_transpose_4/stack/1:output:0+decoder/conv2d_transpose_4/stack/2:output:0+decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
*decoder/conv2d_transpose_4/strided_slice_1StridedSlice)decoder/conv2d_transpose_4/stack:output:09decoder/conv2d_transpose_4/strided_slice_1/stack:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskк
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0┬
+decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_4/stack:output:0Bdecoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
е
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
"decoder/conv2d_transpose_4/BiasAddBiasAdd4decoder/conv2d_transpose_4/conv2d_transpose:output:09decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ј
decoder/conv2d_transpose_4/ReluRelu+decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:         }
 decoder/conv2d_transpose_5/ShapeShape-decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
(decoder/conv2d_transpose_5/strided_sliceStridedSlice)decoder/conv2d_transpose_5/Shape:output:07decoder/conv2d_transpose_5/strided_slice/stack:output:09decoder/conv2d_transpose_5/strided_slice/stack_1:output:09decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :љ
 decoder/conv2d_transpose_5/stackPack1decoder/conv2d_transpose_5/strided_slice:output:0+decoder/conv2d_transpose_5/stack/1:output:0+decoder/conv2d_transpose_5/stack/2:output:0+decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
*decoder/conv2d_transpose_5/strided_slice_1StridedSlice)decoder/conv2d_transpose_5/stack:output:09decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskк
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0┬
+decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_5/stack:output:0Bdecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
е
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
"decoder/conv2d_transpose_5/BiasAddBiasAdd4decoder/conv2d_transpose_5/conv2d_transpose:output:09decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ѓ
IdentityIdentity+decoder/conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         Х
NoOpNoOp2^decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp'^decoder/dense_1/BiasAdd/ReadVariableOp&^decoder/dense_1/MatMul/ReadVariableOp(^encoder/conv2d_4/BiasAdd/ReadVariableOp'^encoder/conv2d_4/Conv2D/ReadVariableOp(^encoder/conv2d_5/BiasAdd/ReadVariableOp'^encoder/conv2d_5/Conv2D/ReadVariableOp(^encoder/conv2d_6/BiasAdd/ReadVariableOp'^encoder/conv2d_6/Conv2D/ReadVariableOp(^encoder/conv2d_7/BiasAdd/ReadVariableOp'^encoder/conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2f
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2P
&decoder/dense_1/BiasAdd/ReadVariableOp&decoder/dense_1/BiasAdd/ReadVariableOp2N
%decoder/dense_1/MatMul/ReadVariableOp%decoder/dense_1/MatMul/ReadVariableOp2R
'encoder/conv2d_4/BiasAdd/ReadVariableOp'encoder/conv2d_4/BiasAdd/ReadVariableOp2P
&encoder/conv2d_4/Conv2D/ReadVariableOp&encoder/conv2d_4/Conv2D/ReadVariableOp2R
'encoder/conv2d_5/BiasAdd/ReadVariableOp'encoder/conv2d_5/BiasAdd/ReadVariableOp2P
&encoder/conv2d_5/Conv2D/ReadVariableOp&encoder/conv2d_5/Conv2D/ReadVariableOp2R
'encoder/conv2d_6/BiasAdd/ReadVariableOp'encoder/conv2d_6/BiasAdd/ReadVariableOp2P
&encoder/conv2d_6/Conv2D/ReadVariableOp&encoder/conv2d_6/Conv2D/ReadVariableOp2R
'encoder/conv2d_7/BiasAdd/ReadVariableOp'encoder/conv2d_7/BiasAdd/ReadVariableOp2P
&encoder/conv2d_7/Conv2D/ReadVariableOp&encoder/conv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Т	
Л
(__inference_decoder_layer_call_fn_366135

inputs
unknown:	љ
	unknown_0:	љ#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identityѕбStatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_365180w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
№	
п
(__inference_encoder_layer_call_fn_364931
input_3!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@
	unknown_6:
identityѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_364891o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3
╦
Я
C__inference_model_1_layer_call_and_return_conditional_losses_365395

inputs(
encoder_365360:
encoder_365362:(
encoder_365364: 
encoder_365366: (
encoder_365368: @
encoder_365370:@(
encoder_365372:@
encoder_365374:!
decoder_365377:	љ
decoder_365379:	љ(
decoder_365381:
decoder_365383:(
decoder_365385:
decoder_365387:(
decoder_365389:
decoder_365391:
identityѕбdecoder/StatefulPartitionedCallбencoder/StatefulPartitionedCallп
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_365360encoder_365362encoder_365364encoder_365366encoder_365368encoder_365370encoder_365372encoder_365374*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_364781ѓ
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_365377decoder_365379decoder_365381decoder_365383decoder_365385decoder_365387decoder_365389decoder_365391*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_365180
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         і
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╬
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_366473

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Е
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         љ:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
В	
О
(__inference_encoder_layer_call_fn_366019

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@
	unknown_6:
identityѕбStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_364781o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
­
╬
(__inference_model_1_layer_call_fn_365737

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@
	unknown_6:
	unknown_7:	љ
	unknown_8:	љ#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_365395w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ђe
ё
C__inference_decoder_layer_call_and_return_conditional_losses_366235

inputs9
&dense_1_matmul_readvariableop_resource:	љ6
'dense_1_biasadd_readvariableop_resource:	љU
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_3_biasadd_readvariableop_resource:U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_4_biasadd_readvariableop_resource:U
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_5_biasadd_readvariableop_resource:
identityѕб)conv2d_transpose_3/BiasAdd/ReadVariableOpб2conv2d_transpose_3/conv2d_transpose/ReadVariableOpб)conv2d_transpose_4/BiasAdd/ReadVariableOpб2conv2d_transpose_4/conv2d_transpose/ReadVariableOpб)conv2d_transpose_5/BiasAdd/ReadVariableOpб2conv2d_transpose_5/conv2d_transpose/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpЁ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0z
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЃ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ј
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љW
reshape_1/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :█
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:њ
reshape_1/ReshapeReshapedense_1/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         b
conv2d_transpose_3/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0ў
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
ў
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:         m
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0б
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~
conv2d_transpose_4/ReluRelu#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:         m
conv2d_transpose_5/ShapeShape%conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0б
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         ф
NoOpNoOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ў
W
;__inference_global_average_pooling2d_1_layer_call_fn_366429

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *_
fZRX
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_364698i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
К
е
3__inference_conv2d_transpose_3_layer_call_fn_366482

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_365029Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
─
Ќ
(__inference_dense_1_layer_call_fn_366444

inputs
unknown:	љ
	unknown_0:	љ
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_365142p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         љ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ж
ъ
)__inference_conv2d_5_layer_call_fn_366353

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_364737w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
В	
О
(__inference_encoder_layer_call_fn_366040

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@
	unknown_6:
identityѕбStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_364891o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
▒
F
*__inference_reshape_1_layer_call_fn_366459

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_365162h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         љ:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
Т	
Л
(__inference_decoder_layer_call_fn_366156

inputs
unknown:	љ
	unknown_0:	љ#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identityѕбStatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_365263w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
а#
Ќ
C__inference_encoder_layer_call_and_return_conditional_losses_364959
input_3)
conv2d_4_364934:
conv2d_4_364936:)
conv2d_5_364940: 
conv2d_5_364942: )
conv2d_6_364946: @
conv2d_6_364948:@)
conv2d_7_364952:@
conv2d_7_364954:
identityѕб conv2d_4/StatefulPartitionedCallб conv2d_5/StatefulPartitionedCallб conv2d_6/StatefulPartitionedCallб conv2d_7/StatefulPartitionedCallщ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_4_364934conv2d_4_364936*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_364719ы
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_364661џ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_5_364940conv2d_5_364942*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_364737ы
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_364673џ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_6_364946conv2d_6_364948*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_364755ы
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_364685џ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_7_364952conv2d_7_364954*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_364773 
*global_average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *_
fZRX
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_364698ѓ
IdentityIdentity3global_average_pooling2d_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         м
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3
Ь-
┌
C__inference_encoder_layer_call_and_return_conditional_losses_366077

inputsA
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:A
'conv2d_5_conv2d_readvariableop_resource: 6
(conv2d_5_biasadd_readvariableop_resource: A
'conv2d_6_conv2d_readvariableop_resource: @6
(conv2d_6_biasadd_readvariableop_resource:@A
'conv2d_7_conv2d_readvariableop_resource:@6
(conv2d_7_biasadd_readvariableop_resource:
identityѕбconv2d_4/BiasAdd/ReadVariableOpбconv2d_4/Conv2D/ReadVariableOpбconv2d_5/BiasAdd/ReadVariableOpбconv2d_5/Conv2D/ReadVariableOpбconv2d_6/BiasAdd/ReadVariableOpбconv2d_6/Conv2D/ReadVariableOpбconv2d_7/BiasAdd/ReadVariableOpбconv2d_7/Conv2D/ReadVariableOpј
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ё
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         j
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         г
max_pooling2d_3/MaxPoolMaxPoolconv2d_4/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
ј
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0┼
conv2d_5/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ё
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          j
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:          г
max_pooling2d_4/MaxPoolMaxPoolconv2d_5/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
ј
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0┼
conv2d_6/Conv2DConv2D max_pooling2d_4/MaxPool:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ё
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ў
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         @г
max_pooling2d_5/MaxPoolMaxPoolconv2d_6/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ј
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0┼
conv2d_7/Conv2DConv2D max_pooling2d_5/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ё
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         ѓ
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ▓
global_average_pooling2d_1/MeanMeanconv2d_7/Relu:activations:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         w
IdentityIdentity(global_average_pooling2d_1/Mean:output:0^NoOp*
T0*'
_output_shapes
:         м
NoOpNoOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
з
¤
(__inference_model_1_layer_call_fn_365430
input_3!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@
	unknown_6:
	unknown_7:	љ
	unknown_8:	љ#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_365395w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3
├
у
C__inference_decoder_layer_call_and_return_conditional_losses_365180

inputs!
dense_1_365143:	љ
dense_1_365145:	љ3
conv2d_transpose_3_365164:'
conv2d_transpose_3_365166:3
conv2d_transpose_4_365169:'
conv2d_transpose_4_365171:3
conv2d_transpose_5_365174:'
conv2d_transpose_5_365176:
identityѕб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCallб*conv2d_transpose_5/StatefulPartitionedCallбdense_1/StatefulPartitionedCallь
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_365143dense_1_365145*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_365142С
reshape_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_365162╝
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_3_365164conv2d_transpose_3_365166*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_365029═
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_365169conv2d_transpose_4_365171*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_365074═
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_365174conv2d_transpose_5_365176*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_365118і
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         №
NoOpNoOp+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_366344

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
И
L
0__inference_max_pooling2d_5_layer_call_fn_366399

inputs
identity┘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_364685Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
и
r
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_366435

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╦
Я
C__inference_model_1_layer_call_and_return_conditional_losses_365507

inputs(
encoder_365472:
encoder_365474:(
encoder_365476: 
encoder_365478: (
encoder_365480: @
encoder_365482:@(
encoder_365484:@
encoder_365486:!
decoder_365489:	љ
decoder_365491:	љ(
decoder_365493:
decoder_365495:(
decoder_365497:
decoder_365499:(
decoder_365501:
decoder_365503:
identityѕбdecoder/StatefulPartitionedCallбencoder/StatefulPartitionedCallп
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_365472encoder_365474encoder_365476encoder_365478encoder_365480encoder_365482encoder_365484encoder_365486*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_364891ѓ
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_365489decoder_365491decoder_365493decoder_365495decoder_365497decoder_365499decoder_365501decoder_365503*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_365263
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         і
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ж
ъ
)__inference_conv2d_7_layer_call_fn_366413

inputs!
unknown:@
	unknown_0:
identityѕбStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_364773w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_364685

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ѓ
§
D__inference_conv2d_4_layer_call_and_return_conditional_losses_364719

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_364673

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
к
У
C__inference_decoder_layer_call_and_return_conditional_losses_365328
input_4!
dense_1_365306:	љ
dense_1_365308:	љ3
conv2d_transpose_3_365312:'
conv2d_transpose_3_365314:3
conv2d_transpose_4_365317:'
conv2d_transpose_4_365319:3
conv2d_transpose_5_365322:'
conv2d_transpose_5_365324:
identityѕб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCallб*conv2d_transpose_5/StatefulPartitionedCallбdense_1/StatefulPartitionedCallЬ
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_1_365306dense_1_365308*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_365142С
reshape_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_365162╝
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_3_365312conv2d_transpose_3_365314*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_365029═
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_365317conv2d_transpose_4_365319*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_365074═
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_365322conv2d_transpose_5_365324*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_365118і
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         №
NoOpNoOp+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_4
И
L
0__inference_max_pooling2d_4_layer_call_fn_366369

inputs
identity┘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_364673Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
═	
Ш
C__inference_dense_1_layer_call_and_return_conditional_losses_365142

inputs1
matmul_readvariableop_resource:	љ.
biasadd_readvariableop_resource:	љ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         љw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
эС
ш%
"__inference__traced_restore_366986
file_prefix:
 assignvariableop_conv2d_4_kernel:.
 assignvariableop_1_conv2d_4_bias:<
"assignvariableop_2_conv2d_5_kernel: .
 assignvariableop_3_conv2d_5_bias: <
"assignvariableop_4_conv2d_6_kernel: @.
 assignvariableop_5_conv2d_6_bias:@<
"assignvariableop_6_conv2d_7_kernel:@.
 assignvariableop_7_conv2d_7_bias:4
!assignvariableop_8_dense_1_kernel:	љ.
assignvariableop_9_dense_1_bias:	љG
-assignvariableop_10_conv2d_transpose_3_kernel:9
+assignvariableop_11_conv2d_transpose_3_bias:G
-assignvariableop_12_conv2d_transpose_4_kernel:9
+assignvariableop_13_conv2d_transpose_4_bias:G
-assignvariableop_14_conv2d_transpose_5_kernel:9
+assignvariableop_15_conv2d_transpose_5_bias:(
assignvariableop_16_nadam_iter:	 *
 assignvariableop_17_nadam_beta_1: *
 assignvariableop_18_nadam_beta_2: )
assignvariableop_19_nadam_decay: 1
'assignvariableop_20_nadam_learning_rate: 2
(assignvariableop_21_nadam_momentum_cache: %
assignvariableop_22_total_1: %
assignvariableop_23_count_1: #
assignvariableop_24_total: #
assignvariableop_25_count: E
+assignvariableop_26_nadam_conv2d_4_kernel_m:7
)assignvariableop_27_nadam_conv2d_4_bias_m:E
+assignvariableop_28_nadam_conv2d_5_kernel_m: 7
)assignvariableop_29_nadam_conv2d_5_bias_m: E
+assignvariableop_30_nadam_conv2d_6_kernel_m: @7
)assignvariableop_31_nadam_conv2d_6_bias_m:@E
+assignvariableop_32_nadam_conv2d_7_kernel_m:@7
)assignvariableop_33_nadam_conv2d_7_bias_m:=
*assignvariableop_34_nadam_dense_1_kernel_m:	љ7
(assignvariableop_35_nadam_dense_1_bias_m:	љO
5assignvariableop_36_nadam_conv2d_transpose_3_kernel_m:A
3assignvariableop_37_nadam_conv2d_transpose_3_bias_m:O
5assignvariableop_38_nadam_conv2d_transpose_4_kernel_m:A
3assignvariableop_39_nadam_conv2d_transpose_4_bias_m:O
5assignvariableop_40_nadam_conv2d_transpose_5_kernel_m:A
3assignvariableop_41_nadam_conv2d_transpose_5_bias_m:E
+assignvariableop_42_nadam_conv2d_4_kernel_v:7
)assignvariableop_43_nadam_conv2d_4_bias_v:E
+assignvariableop_44_nadam_conv2d_5_kernel_v: 7
)assignvariableop_45_nadam_conv2d_5_bias_v: E
+assignvariableop_46_nadam_conv2d_6_kernel_v: @7
)assignvariableop_47_nadam_conv2d_6_bias_v:@E
+assignvariableop_48_nadam_conv2d_7_kernel_v:@7
)assignvariableop_49_nadam_conv2d_7_bias_v:=
*assignvariableop_50_nadam_dense_1_kernel_v:	љ7
(assignvariableop_51_nadam_dense_1_bias_v:	љO
5assignvariableop_52_nadam_conv2d_transpose_3_kernel_v:A
3assignvariableop_53_nadam_conv2d_transpose_3_bias_v:O
5assignvariableop_54_nadam_conv2d_transpose_4_kernel_v:A
3assignvariableop_55_nadam_conv2d_transpose_4_bias_v:O
5assignvariableop_56_nadam_conv2d_transpose_5_kernel_v:A
3assignvariableop_57_nadam_conv2d_transpose_5_bias_v:
identity_59ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Љ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*и
valueГBф;B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHУ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*і
valueђB~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╚
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ѓ
_output_shapes№
В:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*I
dtypes?
=2;	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOpAssignVariableOp assignvariableop_conv2d_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_10AssignVariableOp-assignvariableop_10_conv2d_transpose_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_11AssignVariableOp+assignvariableop_11_conv2d_transpose_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_12AssignVariableOp-assignvariableop_12_conv2d_transpose_4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_13AssignVariableOp+assignvariableop_13_conv2d_transpose_4_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_14AssignVariableOp-assignvariableop_14_conv2d_transpose_5_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_15AssignVariableOp+assignvariableop_15_conv2d_transpose_5_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:Ј
AssignVariableOp_16AssignVariableOpassignvariableop_16_nadam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_17AssignVariableOp assignvariableop_17_nadam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_18AssignVariableOp assignvariableop_18_nadam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_19AssignVariableOpassignvariableop_19_nadam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_20AssignVariableOp'assignvariableop_20_nadam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_21AssignVariableOp(assignvariableop_21_nadam_momentum_cacheIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_24AssignVariableOpassignvariableop_24_totalIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_25AssignVariableOpassignvariableop_25_countIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_26AssignVariableOp+assignvariableop_26_nadam_conv2d_4_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_27AssignVariableOp)assignvariableop_27_nadam_conv2d_4_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_28AssignVariableOp+assignvariableop_28_nadam_conv2d_5_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_29AssignVariableOp)assignvariableop_29_nadam_conv2d_5_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_30AssignVariableOp+assignvariableop_30_nadam_conv2d_6_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_31AssignVariableOp)assignvariableop_31_nadam_conv2d_6_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_32AssignVariableOp+assignvariableop_32_nadam_conv2d_7_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_33AssignVariableOp)assignvariableop_33_nadam_conv2d_7_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_34AssignVariableOp*assignvariableop_34_nadam_dense_1_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_35AssignVariableOp(assignvariableop_35_nadam_dense_1_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_36AssignVariableOp5assignvariableop_36_nadam_conv2d_transpose_3_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_37AssignVariableOp3assignvariableop_37_nadam_conv2d_transpose_3_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_38AssignVariableOp5assignvariableop_38_nadam_conv2d_transpose_4_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_39AssignVariableOp3assignvariableop_39_nadam_conv2d_transpose_4_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_40AssignVariableOp5assignvariableop_40_nadam_conv2d_transpose_5_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_41AssignVariableOp3assignvariableop_41_nadam_conv2d_transpose_5_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_42AssignVariableOp+assignvariableop_42_nadam_conv2d_4_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_43AssignVariableOp)assignvariableop_43_nadam_conv2d_4_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_44AssignVariableOp+assignvariableop_44_nadam_conv2d_5_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_45AssignVariableOp)assignvariableop_45_nadam_conv2d_5_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_46AssignVariableOp+assignvariableop_46_nadam_conv2d_6_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_47AssignVariableOp)assignvariableop_47_nadam_conv2d_6_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_48AssignVariableOp+assignvariableop_48_nadam_conv2d_7_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_49AssignVariableOp)assignvariableop_49_nadam_conv2d_7_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_50AssignVariableOp*assignvariableop_50_nadam_dense_1_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_51AssignVariableOp(assignvariableop_51_nadam_dense_1_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_52AssignVariableOp5assignvariableop_52_nadam_conv2d_transpose_3_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_53AssignVariableOp3assignvariableop_53_nadam_conv2d_transpose_3_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_54AssignVariableOp5assignvariableop_54_nadam_conv2d_transpose_4_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_55AssignVariableOp3assignvariableop_55_nadam_conv2d_transpose_4_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_56AssignVariableOp5assignvariableop_56_nadam_conv2d_transpose_5_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_57AssignVariableOp3assignvariableop_57_nadam_conv2d_transpose_5_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ╦

Identity_58Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_59IdentityIdentity_58:output:0^NoOp_1*
T0*
_output_shapes
: И

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_59Identity_59:output:0*Ѕ
_input_shapesx
v: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_57AssignVariableOp_572(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ю#
ќ
C__inference_encoder_layer_call_and_return_conditional_losses_364781

inputs)
conv2d_4_364720:
conv2d_4_364722:)
conv2d_5_364738: 
conv2d_5_364740: )
conv2d_6_364756: @
conv2d_6_364758:@)
conv2d_7_364774:@
conv2d_7_364776:
identityѕб conv2d_4/StatefulPartitionedCallб conv2d_5/StatefulPartitionedCallб conv2d_6/StatefulPartitionedCallб conv2d_7/StatefulPartitionedCallЭ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_364720conv2d_4_364722*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_364719ы
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_364661џ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_5_364738conv2d_5_364740*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_364737ы
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_364673џ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_6_364756conv2d_6_364758*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_364755ы
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_364685џ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_7_364774conv2d_7_364776*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_364773 
*global_average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *_
fZRX
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_364698ѓ
IdentityIdentity3global_average_pooling2d_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         м
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
о 
Џ
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_365118

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
value	B :y
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ю#
ќ
C__inference_encoder_layer_call_and_return_conditional_losses_364891

inputs)
conv2d_4_364866:
conv2d_4_364868:)
conv2d_5_364872: 
conv2d_5_364874: )
conv2d_6_364878: @
conv2d_6_364880:@)
conv2d_7_364884:@
conv2d_7_364886:
identityѕб conv2d_4/StatefulPartitionedCallб conv2d_5/StatefulPartitionedCallб conv2d_6/StatefulPartitionedCallб conv2d_7/StatefulPartitionedCallЭ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_364866conv2d_4_364868*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_364719ы
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_364661џ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_5_364872conv2d_5_364874*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_364737ы
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_364673џ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_6_364878conv2d_6_364880*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_364755ы
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_364685џ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_7_364884conv2d_7_364886*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_364773 
*global_average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *_
fZRX
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_364698ѓ
IdentityIdentity3global_average_pooling2d_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         м
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
дq
а
__inference__traced_save_366802
file_prefix.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop8
4savev2_conv2d_transpose_5_kernel_read_readvariableop6
2savev2_conv2d_transpose_5_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_nadam_conv2d_4_kernel_m_read_readvariableop4
0savev2_nadam_conv2d_4_bias_m_read_readvariableop6
2savev2_nadam_conv2d_5_kernel_m_read_readvariableop4
0savev2_nadam_conv2d_5_bias_m_read_readvariableop6
2savev2_nadam_conv2d_6_kernel_m_read_readvariableop4
0savev2_nadam_conv2d_6_bias_m_read_readvariableop6
2savev2_nadam_conv2d_7_kernel_m_read_readvariableop4
0savev2_nadam_conv2d_7_bias_m_read_readvariableop5
1savev2_nadam_dense_1_kernel_m_read_readvariableop3
/savev2_nadam_dense_1_bias_m_read_readvariableop@
<savev2_nadam_conv2d_transpose_3_kernel_m_read_readvariableop>
:savev2_nadam_conv2d_transpose_3_bias_m_read_readvariableop@
<savev2_nadam_conv2d_transpose_4_kernel_m_read_readvariableop>
:savev2_nadam_conv2d_transpose_4_bias_m_read_readvariableop@
<savev2_nadam_conv2d_transpose_5_kernel_m_read_readvariableop>
:savev2_nadam_conv2d_transpose_5_bias_m_read_readvariableop6
2savev2_nadam_conv2d_4_kernel_v_read_readvariableop4
0savev2_nadam_conv2d_4_bias_v_read_readvariableop6
2savev2_nadam_conv2d_5_kernel_v_read_readvariableop4
0savev2_nadam_conv2d_5_bias_v_read_readvariableop6
2savev2_nadam_conv2d_6_kernel_v_read_readvariableop4
0savev2_nadam_conv2d_6_bias_v_read_readvariableop6
2savev2_nadam_conv2d_7_kernel_v_read_readvariableop4
0savev2_nadam_conv2d_7_bias_v_read_readvariableop5
1savev2_nadam_dense_1_kernel_v_read_readvariableop3
/savev2_nadam_dense_1_bias_v_read_readvariableop@
<savev2_nadam_conv2d_transpose_3_kernel_v_read_readvariableop>
:savev2_nadam_conv2d_transpose_3_bias_v_read_readvariableop@
<savev2_nadam_conv2d_transpose_4_kernel_v_read_readvariableop>
:savev2_nadam_conv2d_transpose_4_bias_v_read_readvariableop@
<savev2_nadam_conv2d_transpose_5_kernel_v_read_readvariableop>
:savev2_nadam_conv2d_transpose_5_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ј
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*и
valueГBф;B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHт
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*і
valueђB~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Х
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop4savev2_conv2d_transpose_5_kernel_read_readvariableop2savev2_conv2d_transpose_5_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_nadam_conv2d_4_kernel_m_read_readvariableop0savev2_nadam_conv2d_4_bias_m_read_readvariableop2savev2_nadam_conv2d_5_kernel_m_read_readvariableop0savev2_nadam_conv2d_5_bias_m_read_readvariableop2savev2_nadam_conv2d_6_kernel_m_read_readvariableop0savev2_nadam_conv2d_6_bias_m_read_readvariableop2savev2_nadam_conv2d_7_kernel_m_read_readvariableop0savev2_nadam_conv2d_7_bias_m_read_readvariableop1savev2_nadam_dense_1_kernel_m_read_readvariableop/savev2_nadam_dense_1_bias_m_read_readvariableop<savev2_nadam_conv2d_transpose_3_kernel_m_read_readvariableop:savev2_nadam_conv2d_transpose_3_bias_m_read_readvariableop<savev2_nadam_conv2d_transpose_4_kernel_m_read_readvariableop:savev2_nadam_conv2d_transpose_4_bias_m_read_readvariableop<savev2_nadam_conv2d_transpose_5_kernel_m_read_readvariableop:savev2_nadam_conv2d_transpose_5_bias_m_read_readvariableop2savev2_nadam_conv2d_4_kernel_v_read_readvariableop0savev2_nadam_conv2d_4_bias_v_read_readvariableop2savev2_nadam_conv2d_5_kernel_v_read_readvariableop0savev2_nadam_conv2d_5_bias_v_read_readvariableop2savev2_nadam_conv2d_6_kernel_v_read_readvariableop0savev2_nadam_conv2d_6_bias_v_read_readvariableop2savev2_nadam_conv2d_7_kernel_v_read_readvariableop0savev2_nadam_conv2d_7_bias_v_read_readvariableop1savev2_nadam_dense_1_kernel_v_read_readvariableop/savev2_nadam_dense_1_bias_v_read_readvariableop<savev2_nadam_conv2d_transpose_3_kernel_v_read_readvariableop:savev2_nadam_conv2d_transpose_3_bias_v_read_readvariableop<savev2_nadam_conv2d_transpose_4_kernel_v_read_readvariableop:savev2_nadam_conv2d_transpose_4_bias_v_read_readvariableop<savev2_nadam_conv2d_transpose_5_kernel_v_read_readvariableop:savev2_nadam_conv2d_transpose_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *I
dtypes?
=2;	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
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

identity_1Identity_1:output:0*█
_input_shapes╔
к: ::: : : @:@:@::	љ:љ::::::: : : : : : : : : : ::: : : @:@:@::	љ:љ::::::::: : : @:@:@::	љ:љ::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
::%	!

_output_shapes
:	љ:!


_output_shapes	
:љ:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @:  

_output_shapes
:@:,!(
&
_output_shapes
:@: "

_output_shapes
::%#!

_output_shapes
:	љ:!$

_output_shapes	
:љ:,%(
&
_output_shapes
:: &

_output_shapes
::,'(
&
_output_shapes
:: (

_output_shapes
::,)(
&
_output_shapes
:: *

_output_shapes
::,+(
&
_output_shapes
:: ,

_output_shapes
::,-(
&
_output_shapes
: : .

_output_shapes
: :,/(
&
_output_shapes
: @: 0

_output_shapes
:@:,1(
&
_output_shapes
:@: 2

_output_shapes
::%3!

_output_shapes
:	љ:!4

_output_shapes	
:љ:,5(
&
_output_shapes
:: 6

_output_shapes
::,7(
&
_output_shapes
:: 8

_output_shapes
::,9(
&
_output_shapes
:: :

_output_shapes
::;

_output_shapes
: 
Ѓ
§
D__inference_conv2d_5_layer_call_and_return_conditional_losses_366364

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
═
╦
$__inference_signature_wrapper_365700
input_3!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@
	unknown_6:
	unknown_7:	љ
	unknown_8:	љ#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_364652w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3
з
¤
(__inference_model_1_layer_call_fn_365579
input_3!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@
	unknown_6:
	unknown_7:	љ
	unknown_8:	љ#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_365507w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3
И
L
0__inference_max_pooling2d_3_layer_call_fn_366339

inputs
identity┘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_364661Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ѓ
§
D__inference_conv2d_4_layer_call_and_return_conditional_losses_366334

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ж	
м
(__inference_decoder_layer_call_fn_365199
input_4
unknown:	љ
	unknown_0:	љ#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identityѕбStatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_365180w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_4
╬
р
C__inference_model_1_layer_call_and_return_conditional_losses_365617
input_3(
encoder_365582:
encoder_365584:(
encoder_365586: 
encoder_365588: (
encoder_365590: @
encoder_365592:@(
encoder_365594:@
encoder_365596:!
decoder_365599:	љ
decoder_365601:	љ(
decoder_365603:
decoder_365605:(
decoder_365607:
decoder_365609:(
decoder_365611:
decoder_365613:
identityѕбdecoder/StatefulPartitionedCallбencoder/StatefulPartitionedCall┘
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_3encoder_365582encoder_365584encoder_365586encoder_365588encoder_365590encoder_365592encoder_365594encoder_365596*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_364781ѓ
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_365599decoder_365601decoder_365603decoder_365605decoder_365607decoder_365609decoder_365611decoder_365613*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_365180
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         і
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3
Ж
ъ
)__inference_conv2d_6_layer_call_fn_366383

inputs!
unknown: @
	unknown_0:@
identityѕбStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_364755w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
├
у
C__inference_decoder_layer_call_and_return_conditional_losses_365263

inputs!
dense_1_365241:	љ
dense_1_365243:	љ3
conv2d_transpose_3_365247:'
conv2d_transpose_3_365249:3
conv2d_transpose_4_365252:'
conv2d_transpose_4_365254:3
conv2d_transpose_5_365257:'
conv2d_transpose_5_365259:
identityѕб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCallб*conv2d_transpose_5/StatefulPartitionedCallбdense_1/StatefulPartitionedCallь
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_365241dense_1_365243*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_365142С
reshape_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_365162╝
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_3_365247conv2d_transpose_3_365249*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_365029═
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_365252conv2d_transpose_4_365254*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_365074═
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_365257conv2d_transpose_5_365259*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_365118і
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         №
NoOpNoOp+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
№	
п
(__inference_encoder_layer_call_fn_364800
input_3!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@
	unknown_6:
identityѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_364781o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3
Њ
g
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_366404

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ж
ъ
)__inference_conv2d_4_layer_call_fn_366323

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_364719w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
К
е
3__inference_conv2d_transpose_4_layer_call_fn_366529

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_365074Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
№#
Џ
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_365029

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
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
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0П
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
№#
Џ
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_366520

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
: G
add/yConst*
_output_shapes
: *
dtype0*
value	B :F
addAddV2mul:z:0add/y:output:0*
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
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :L
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0П
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ѓ
§
D__inference_conv2d_7_layer_call_and_return_conditional_losses_364773

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
о 
Џ
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_366605

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
value	B :y
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+                           Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ђe
ё
C__inference_decoder_layer_call_and_return_conditional_losses_366314

inputs9
&dense_1_matmul_readvariableop_resource:	љ6
'dense_1_biasadd_readvariableop_resource:	љU
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_3_biasadd_readvariableop_resource:U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_4_biasadd_readvariableop_resource:U
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_5_biasadd_readvariableop_resource:
identityѕб)conv2d_transpose_3/BiasAdd/ReadVariableOpб2conv2d_transpose_3/conv2d_transpose/ReadVariableOpб)conv2d_transpose_4/BiasAdd/ReadVariableOpб2conv2d_transpose_4/conv2d_transpose/ReadVariableOpб)conv2d_transpose_5/BiasAdd/ReadVariableOpб2conv2d_transpose_5/conv2d_transpose/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpЁ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0z
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЃ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ј
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љW
reshape_1/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :█
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:њ
reshape_1/ReshapeReshapedense_1/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         b
conv2d_transpose_3/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0ў
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
ў
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:         m
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0б
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~
conv2d_transpose_4/ReluRelu#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:         m
conv2d_transpose_5/ShapeShape%conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0б
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         ф
NoOpNoOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
─!
Џ
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_366563

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOp;
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
valueB:Л
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
valueB:┘
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
valueB:┘
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
value	B :y
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
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskљ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ў
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           Ђ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ѓ
§
D__inference_conv2d_7_layer_call_and_return_conditional_losses_366424

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
К
е
3__inference_conv2d_transpose_5_layer_call_fn_366572

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_365118Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╬
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_365162

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Е
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         `
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         љ:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
а#
Ќ
C__inference_encoder_layer_call_and_return_conditional_losses_364987
input_3)
conv2d_4_364962:
conv2d_4_364964:)
conv2d_5_364968: 
conv2d_5_364970: )
conv2d_6_364974: @
conv2d_6_364976:@)
conv2d_7_364980:@
conv2d_7_364982:
identityѕб conv2d_4/StatefulPartitionedCallб conv2d_5/StatefulPartitionedCallб conv2d_6/StatefulPartitionedCallб conv2d_7/StatefulPartitionedCallщ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_4_364962conv2d_4_364964*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_364719ы
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_364661џ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_5_364968conv2d_5_364970*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_364737ы
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_364673џ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_6_364974conv2d_6_364976*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_364755ы
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_364685џ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_7_364980conv2d_7_364982*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_364773 
*global_average_pooling2d_1/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *_
fZRX
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_364698ѓ
IdentityIdentity3global_average_pooling2d_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         м
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3
Ѓ
§
D__inference_conv2d_5_layer_call_and_return_conditional_losses_364737

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ж▓
█
!__inference__wrapped_model_364652
input_3Q
7model_1_encoder_conv2d_4_conv2d_readvariableop_resource:F
8model_1_encoder_conv2d_4_biasadd_readvariableop_resource:Q
7model_1_encoder_conv2d_5_conv2d_readvariableop_resource: F
8model_1_encoder_conv2d_5_biasadd_readvariableop_resource: Q
7model_1_encoder_conv2d_6_conv2d_readvariableop_resource: @F
8model_1_encoder_conv2d_6_biasadd_readvariableop_resource:@Q
7model_1_encoder_conv2d_7_conv2d_readvariableop_resource:@F
8model_1_encoder_conv2d_7_biasadd_readvariableop_resource:I
6model_1_decoder_dense_1_matmul_readvariableop_resource:	љF
7model_1_decoder_dense_1_biasadd_readvariableop_resource:	љe
Kmodel_1_decoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:P
Bmodel_1_decoder_conv2d_transpose_3_biasadd_readvariableop_resource:e
Kmodel_1_decoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:P
Bmodel_1_decoder_conv2d_transpose_4_biasadd_readvariableop_resource:e
Kmodel_1_decoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:P
Bmodel_1_decoder_conv2d_transpose_5_biasadd_readvariableop_resource:
identityѕб9model_1/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpбBmodel_1/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpб9model_1/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpбBmodel_1/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpб9model_1/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpбBmodel_1/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpб.model_1/decoder/dense_1/BiasAdd/ReadVariableOpб-model_1/decoder/dense_1/MatMul/ReadVariableOpб/model_1/encoder/conv2d_4/BiasAdd/ReadVariableOpб.model_1/encoder/conv2d_4/Conv2D/ReadVariableOpб/model_1/encoder/conv2d_5/BiasAdd/ReadVariableOpб.model_1/encoder/conv2d_5/Conv2D/ReadVariableOpб/model_1/encoder/conv2d_6/BiasAdd/ReadVariableOpб.model_1/encoder/conv2d_6/Conv2D/ReadVariableOpб/model_1/encoder/conv2d_7/BiasAdd/ReadVariableOpб.model_1/encoder/conv2d_7/Conv2D/ReadVariableOp«
.model_1/encoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp7model_1_encoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╠
model_1/encoder/conv2d_4/Conv2DConv2Dinput_36model_1/encoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ц
/model_1/encoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp8model_1_encoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╚
 model_1/encoder/conv2d_4/BiasAddBiasAdd(model_1/encoder/conv2d_4/Conv2D:output:07model_1/encoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         і
model_1/encoder/conv2d_4/ReluRelu)model_1/encoder/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         ╠
'model_1/encoder/max_pooling2d_3/MaxPoolMaxPool+model_1/encoder/conv2d_4/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
«
.model_1/encoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp7model_1_encoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ш
model_1/encoder/conv2d_5/Conv2DConv2D0model_1/encoder/max_pooling2d_3/MaxPool:output:06model_1/encoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ц
/model_1/encoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp8model_1_encoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╚
 model_1/encoder/conv2d_5/BiasAddBiasAdd(model_1/encoder/conv2d_5/Conv2D:output:07model_1/encoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          і
model_1/encoder/conv2d_5/ReluRelu)model_1/encoder/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╠
'model_1/encoder/max_pooling2d_4/MaxPoolMaxPool+model_1/encoder/conv2d_5/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
«
.model_1/encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp7model_1_encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ш
model_1/encoder/conv2d_6/Conv2DConv2D0model_1/encoder/max_pooling2d_4/MaxPool:output:06model_1/encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ц
/model_1/encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp8model_1_encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╚
 model_1/encoder/conv2d_6/BiasAddBiasAdd(model_1/encoder/conv2d_6/Conv2D:output:07model_1/encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @і
model_1/encoder/conv2d_6/ReluRelu)model_1/encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         @╠
'model_1/encoder/max_pooling2d_5/MaxPoolMaxPool+model_1/encoder/conv2d_6/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
«
.model_1/encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp7model_1_encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ш
model_1/encoder/conv2d_7/Conv2DConv2D0model_1/encoder/max_pooling2d_5/MaxPool:output:06model_1/encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ц
/model_1/encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp8model_1_encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╚
 model_1/encoder/conv2d_7/BiasAddBiasAdd(model_1/encoder/conv2d_7/Conv2D:output:07model_1/encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         і
model_1/encoder/conv2d_7/ReluRelu)model_1/encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         њ
Amodel_1/encoder/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Р
/model_1/encoder/global_average_pooling2d_1/MeanMean+model_1/encoder/conv2d_7/Relu:activations:0Jmodel_1/encoder/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Ц
-model_1/decoder/dense_1/MatMul/ReadVariableOpReadVariableOp6model_1_decoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0╠
model_1/decoder/dense_1/MatMulMatMul8model_1/encoder/global_average_pooling2d_1/Mean:output:05model_1/decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љБ
.model_1/decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp7model_1_decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0┐
model_1/decoder/dense_1/BiasAddBiasAdd(model_1/decoder/dense_1/MatMul:product:06model_1/decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љw
model_1/decoder/reshape_1/ShapeShape(model_1/decoder/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:w
-model_1/decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model_1/decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model_1/decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:М
'model_1/decoder/reshape_1/strided_sliceStridedSlice(model_1/decoder/reshape_1/Shape:output:06model_1/decoder/reshape_1/strided_slice/stack:output:08model_1/decoder/reshape_1/strided_slice/stack_1:output:08model_1/decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)model_1/decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :k
)model_1/decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :k
)model_1/decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ф
'model_1/decoder/reshape_1/Reshape/shapePack0model_1/decoder/reshape_1/strided_slice:output:02model_1/decoder/reshape_1/Reshape/shape/1:output:02model_1/decoder/reshape_1/Reshape/shape/2:output:02model_1/decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:┬
!model_1/decoder/reshape_1/ReshapeReshape(model_1/decoder/dense_1/BiasAdd:output:00model_1/decoder/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         ѓ
(model_1/decoder/conv2d_transpose_3/ShapeShape*model_1/decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
:ђ
6model_1/decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ѓ
8model_1/decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ѓ
8model_1/decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ђ
0model_1/decoder/conv2d_transpose_3/strided_sliceStridedSlice1model_1/decoder/conv2d_transpose_3/Shape:output:0?model_1/decoder/conv2d_transpose_3/strided_slice/stack:output:0Amodel_1/decoder/conv2d_transpose_3/strided_slice/stack_1:output:0Amodel_1/decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*model_1/decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :l
*model_1/decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :l
*model_1/decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :И
(model_1/decoder/conv2d_transpose_3/stackPack9model_1/decoder/conv2d_transpose_3/strided_slice:output:03model_1/decoder/conv2d_transpose_3/stack/1:output:03model_1/decoder/conv2d_transpose_3/stack/2:output:03model_1/decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:ѓ
8model_1/decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ё
:model_1/decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ё
:model_1/decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
2model_1/decoder/conv2d_transpose_3/strided_slice_1StridedSlice1model_1/decoder/conv2d_transpose_3/stack:output:0Amodel_1/decoder/conv2d_transpose_3/strided_slice_1/stack:output:0Cmodel_1/decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0Cmodel_1/decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskо
Bmodel_1/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpKmodel_1_decoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0п
3model_1/decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput1model_1/decoder/conv2d_transpose_3/stack:output:0Jmodel_1/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0*model_1/decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
И
9model_1/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpBmodel_1_decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0­
*model_1/decoder/conv2d_transpose_3/BiasAddBiasAdd<model_1/decoder/conv2d_transpose_3/conv2d_transpose:output:0Amodel_1/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ъ
'model_1/decoder/conv2d_transpose_3/ReluRelu3model_1/decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:         Ї
(model_1/decoder/conv2d_transpose_4/ShapeShape5model_1/decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:ђ
6model_1/decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ѓ
8model_1/decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ѓ
8model_1/decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ђ
0model_1/decoder/conv2d_transpose_4/strided_sliceStridedSlice1model_1/decoder/conv2d_transpose_4/Shape:output:0?model_1/decoder/conv2d_transpose_4/strided_slice/stack:output:0Amodel_1/decoder/conv2d_transpose_4/strided_slice/stack_1:output:0Amodel_1/decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*model_1/decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :l
*model_1/decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :l
*model_1/decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :И
(model_1/decoder/conv2d_transpose_4/stackPack9model_1/decoder/conv2d_transpose_4/strided_slice:output:03model_1/decoder/conv2d_transpose_4/stack/1:output:03model_1/decoder/conv2d_transpose_4/stack/2:output:03model_1/decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:ѓ
8model_1/decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ё
:model_1/decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ё
:model_1/decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
2model_1/decoder/conv2d_transpose_4/strided_slice_1StridedSlice1model_1/decoder/conv2d_transpose_4/stack:output:0Amodel_1/decoder/conv2d_transpose_4/strided_slice_1/stack:output:0Cmodel_1/decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0Cmodel_1/decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskо
Bmodel_1/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpKmodel_1_decoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Р
3model_1/decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput1model_1/decoder/conv2d_transpose_4/stack:output:0Jmodel_1/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:05model_1/decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
И
9model_1/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOpBmodel_1_decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0­
*model_1/decoder/conv2d_transpose_4/BiasAddBiasAdd<model_1/decoder/conv2d_transpose_4/conv2d_transpose:output:0Amodel_1/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ъ
'model_1/decoder/conv2d_transpose_4/ReluRelu3model_1/decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:         Ї
(model_1/decoder/conv2d_transpose_5/ShapeShape5model_1/decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:ђ
6model_1/decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ѓ
8model_1/decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ѓ
8model_1/decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ђ
0model_1/decoder/conv2d_transpose_5/strided_sliceStridedSlice1model_1/decoder/conv2d_transpose_5/Shape:output:0?model_1/decoder/conv2d_transpose_5/strided_slice/stack:output:0Amodel_1/decoder/conv2d_transpose_5/strided_slice/stack_1:output:0Amodel_1/decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*model_1/decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :l
*model_1/decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :l
*model_1/decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :И
(model_1/decoder/conv2d_transpose_5/stackPack9model_1/decoder/conv2d_transpose_5/strided_slice:output:03model_1/decoder/conv2d_transpose_5/stack/1:output:03model_1/decoder/conv2d_transpose_5/stack/2:output:03model_1/decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:ѓ
8model_1/decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ё
:model_1/decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ё
:model_1/decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
2model_1/decoder/conv2d_transpose_5/strided_slice_1StridedSlice1model_1/decoder/conv2d_transpose_5/stack:output:0Amodel_1/decoder/conv2d_transpose_5/strided_slice_1/stack:output:0Cmodel_1/decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0Cmodel_1/decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskо
Bmodel_1/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpKmodel_1_decoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Р
3model_1/decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput1model_1/decoder/conv2d_transpose_5/stack:output:0Jmodel_1/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:05model_1/decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
И
9model_1/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOpBmodel_1_decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0­
*model_1/decoder/conv2d_transpose_5/BiasAddBiasAdd<model_1/decoder/conv2d_transpose_5/conv2d_transpose:output:0Amodel_1/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         і
IdentityIdentity3model_1/decoder/conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         Х
NoOpNoOp:^model_1/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpC^model_1/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:^model_1/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpC^model_1/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:^model_1/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpC^model_1/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp/^model_1/decoder/dense_1/BiasAdd/ReadVariableOp.^model_1/decoder/dense_1/MatMul/ReadVariableOp0^model_1/encoder/conv2d_4/BiasAdd/ReadVariableOp/^model_1/encoder/conv2d_4/Conv2D/ReadVariableOp0^model_1/encoder/conv2d_5/BiasAdd/ReadVariableOp/^model_1/encoder/conv2d_5/Conv2D/ReadVariableOp0^model_1/encoder/conv2d_6/BiasAdd/ReadVariableOp/^model_1/encoder/conv2d_6/Conv2D/ReadVariableOp0^model_1/encoder/conv2d_7/BiasAdd/ReadVariableOp/^model_1/encoder/conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2v
9model_1/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp9model_1/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2ѕ
Bmodel_1/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpBmodel_1/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2v
9model_1/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp9model_1/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2ѕ
Bmodel_1/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpBmodel_1/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2v
9model_1/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp9model_1/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2ѕ
Bmodel_1/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpBmodel_1/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2`
.model_1/decoder/dense_1/BiasAdd/ReadVariableOp.model_1/decoder/dense_1/BiasAdd/ReadVariableOp2^
-model_1/decoder/dense_1/MatMul/ReadVariableOp-model_1/decoder/dense_1/MatMul/ReadVariableOp2b
/model_1/encoder/conv2d_4/BiasAdd/ReadVariableOp/model_1/encoder/conv2d_4/BiasAdd/ReadVariableOp2`
.model_1/encoder/conv2d_4/Conv2D/ReadVariableOp.model_1/encoder/conv2d_4/Conv2D/ReadVariableOp2b
/model_1/encoder/conv2d_5/BiasAdd/ReadVariableOp/model_1/encoder/conv2d_5/BiasAdd/ReadVariableOp2`
.model_1/encoder/conv2d_5/Conv2D/ReadVariableOp.model_1/encoder/conv2d_5/Conv2D/ReadVariableOp2b
/model_1/encoder/conv2d_6/BiasAdd/ReadVariableOp/model_1/encoder/conv2d_6/BiasAdd/ReadVariableOp2`
.model_1/encoder/conv2d_6/Conv2D/ReadVariableOp.model_1/encoder/conv2d_6/Conv2D/ReadVariableOp2b
/model_1/encoder/conv2d_7/BiasAdd/ReadVariableOp/model_1/encoder/conv2d_7/BiasAdd/ReadVariableOp2`
.model_1/encoder/conv2d_7/Conv2D/ReadVariableOp.model_1/encoder/conv2d_7/Conv2D/ReadVariableOp:X T
/
_output_shapes
:         
!
_user_specified_name	input_3
═	
Ш
C__inference_dense_1_layer_call_and_return_conditional_losses_366454

inputs1
matmul_readvariableop_resource:	љ.
biasadd_readvariableop_resource:	љ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         љw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к
У
C__inference_decoder_layer_call_and_return_conditional_losses_365353
input_4!
dense_1_365331:	љ
dense_1_365333:	љ3
conv2d_transpose_3_365337:'
conv2d_transpose_3_365339:3
conv2d_transpose_4_365342:'
conv2d_transpose_4_365344:3
conv2d_transpose_5_365347:'
conv2d_transpose_5_365349:
identityѕб*conv2d_transpose_3/StatefulPartitionedCallб*conv2d_transpose_4/StatefulPartitionedCallб*conv2d_transpose_5/StatefulPartitionedCallбdense_1/StatefulPartitionedCallЬ
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_1_365331dense_1_365333*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_365142С
reshape_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_365162╝
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_3_365337conv2d_transpose_3_365339*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_365029═
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_365342conv2d_transpose_4_365344*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_365074═
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_365347conv2d_transpose_5_365349*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_365118і
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         №
NoOpNoOp+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_4
Њ
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_364661

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
и
r
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_364698

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╬
р
C__inference_model_1_layer_call_and_return_conditional_losses_365655
input_3(
encoder_365620:
encoder_365622:(
encoder_365624: 
encoder_365626: (
encoder_365628: @
encoder_365630:@(
encoder_365632:@
encoder_365634:!
decoder_365637:	љ
decoder_365639:	љ(
decoder_365641:
decoder_365643:(
decoder_365645:
decoder_365647:(
decoder_365649:
decoder_365651:
identityѕбdecoder/StatefulPartitionedCallбencoder/StatefulPartitionedCall┘
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_3encoder_365620encoder_365622encoder_365624encoder_365626encoder_365628encoder_365630encoder_365632encoder_365634*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_364891ѓ
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_365637decoder_365639decoder_365641decoder_365643decoder_365645decoder_365647decoder_365649decoder_365651*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_365263
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         і
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_3
­
╬
(__inference_model_1_layer_call_fn_365774

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@
	unknown_6:
	unknown_7:	љ
	unknown_8:	љ#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_365507w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
їА
Ч
C__inference_model_1_layer_call_and_return_conditional_losses_365998

inputsI
/encoder_conv2d_4_conv2d_readvariableop_resource:>
0encoder_conv2d_4_biasadd_readvariableop_resource:I
/encoder_conv2d_5_conv2d_readvariableop_resource: >
0encoder_conv2d_5_biasadd_readvariableop_resource: I
/encoder_conv2d_6_conv2d_readvariableop_resource: @>
0encoder_conv2d_6_biasadd_readvariableop_resource:@I
/encoder_conv2d_7_conv2d_readvariableop_resource:@>
0encoder_conv2d_7_biasadd_readvariableop_resource:A
.decoder_dense_1_matmul_readvariableop_resource:	љ>
/decoder_dense_1_biasadd_readvariableop_resource:	љ]
Cdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_3_biasadd_readvariableop_resource:]
Cdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_4_biasadd_readvariableop_resource:]
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_5_biasadd_readvariableop_resource:
identityѕб1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpб:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpб1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpб:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpб1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpб:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpб&decoder/dense_1/BiasAdd/ReadVariableOpб%decoder/dense_1/MatMul/ReadVariableOpб'encoder/conv2d_4/BiasAdd/ReadVariableOpб&encoder/conv2d_4/Conv2D/ReadVariableOpб'encoder/conv2d_5/BiasAdd/ReadVariableOpб&encoder/conv2d_5/Conv2D/ReadVariableOpб'encoder/conv2d_6/BiasAdd/ReadVariableOpб&encoder/conv2d_6/Conv2D/ReadVariableOpб'encoder/conv2d_7/BiasAdd/ReadVariableOpб&encoder/conv2d_7/Conv2D/ReadVariableOpъ
&encoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╗
encoder/conv2d_4/Conv2DConv2Dinputs.encoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ћ
'encoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0░
encoder/conv2d_4/BiasAddBiasAdd encoder/conv2d_4/Conv2D:output:0/encoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
encoder/conv2d_4/ReluRelu!encoder/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         ╝
encoder/max_pooling2d_3/MaxPoolMaxPool#encoder/conv2d_4/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
ъ
&encoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0П
encoder/conv2d_5/Conv2DConv2D(encoder/max_pooling2d_3/MaxPool:output:0.encoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ћ
'encoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0░
encoder/conv2d_5/BiasAddBiasAdd encoder/conv2d_5/Conv2D:output:0/encoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          z
encoder/conv2d_5/ReluRelu!encoder/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
encoder/max_pooling2d_4/MaxPoolMaxPool#encoder/conv2d_5/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
ъ
&encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0П
encoder/conv2d_6/Conv2DConv2D(encoder/max_pooling2d_4/MaxPool:output:0.encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ћ
'encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0░
encoder/conv2d_6/BiasAddBiasAdd encoder/conv2d_6/Conv2D:output:0/encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @z
encoder/conv2d_6/ReluRelu!encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         @╝
encoder/max_pooling2d_5/MaxPoolMaxPool#encoder/conv2d_6/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ъ
&encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0П
encoder/conv2d_7/Conv2DConv2D(encoder/max_pooling2d_5/MaxPool:output:0.encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ћ
'encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0░
encoder/conv2d_7/BiasAddBiasAdd encoder/conv2d_7/Conv2D:output:0/encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
encoder/conv2d_7/ReluRelu!encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         і
9encoder/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ╩
'encoder/global_average_pooling2d_1/MeanMean#encoder/conv2d_7/Relu:activations:0Bencoder/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Ћ
%decoder/dense_1/MatMul/ReadVariableOpReadVariableOp.decoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0┤
decoder/dense_1/MatMulMatMul0encoder/global_average_pooling2d_1/Mean:output:0-decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЊ
&decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Д
decoder/dense_1/BiasAddBiasAdd decoder/dense_1/MatMul:product:0.decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љg
decoder/reshape_1/ShapeShape decoder/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:o
%decoder/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'decoder/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'decoder/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
decoder/reshape_1/strided_sliceStridedSlice decoder/reshape_1/Shape:output:0.decoder/reshape_1/strided_slice/stack:output:00decoder/reshape_1/strided_slice/stack_1:output:00decoder/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!decoder/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!decoder/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!decoder/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ѓ
decoder/reshape_1/Reshape/shapePack(decoder/reshape_1/strided_slice:output:0*decoder/reshape_1/Reshape/shape/1:output:0*decoder/reshape_1/Reshape/shape/2:output:0*decoder/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ф
decoder/reshape_1/ReshapeReshape decoder/dense_1/BiasAdd:output:0(decoder/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:         r
 decoder/conv2d_transpose_3/ShapeShape"decoder/reshape_1/Reshape:output:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
(decoder/conv2d_transpose_3/strided_sliceStridedSlice)decoder/conv2d_transpose_3/Shape:output:07decoder/conv2d_transpose_3/strided_slice/stack:output:09decoder/conv2d_transpose_3/strided_slice/stack_1:output:09decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :љ
 decoder/conv2d_transpose_3/stackPack1decoder/conv2d_transpose_3/strided_slice:output:0+decoder/conv2d_transpose_3/stack/1:output:0+decoder/conv2d_transpose_3/stack/2:output:0+decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice)decoder/conv2d_transpose_3/stack:output:09decoder/conv2d_transpose_3/strided_slice_1/stack:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskк
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0И
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_3/stack:output:0Bdecoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0"decoder/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
е
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
"decoder/conv2d_transpose_3/BiasAddBiasAdd4decoder/conv2d_transpose_3/conv2d_transpose:output:09decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ј
decoder/conv2d_transpose_3/ReluRelu+decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:         }
 decoder/conv2d_transpose_4/ShapeShape-decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
(decoder/conv2d_transpose_4/strided_sliceStridedSlice)decoder/conv2d_transpose_4/Shape:output:07decoder/conv2d_transpose_4/strided_slice/stack:output:09decoder/conv2d_transpose_4/strided_slice/stack_1:output:09decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :љ
 decoder/conv2d_transpose_4/stackPack1decoder/conv2d_transpose_4/strided_slice:output:0+decoder/conv2d_transpose_4/stack/1:output:0+decoder/conv2d_transpose_4/stack/2:output:0+decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
*decoder/conv2d_transpose_4/strided_slice_1StridedSlice)decoder/conv2d_transpose_4/stack:output:09decoder/conv2d_transpose_4/strided_slice_1/stack:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskк
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0┬
+decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_4/stack:output:0Bdecoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
е
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
"decoder/conv2d_transpose_4/BiasAddBiasAdd4decoder/conv2d_transpose_4/conv2d_transpose:output:09decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ј
decoder/conv2d_transpose_4/ReluRelu+decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:         }
 decoder/conv2d_transpose_5/ShapeShape-decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
(decoder/conv2d_transpose_5/strided_sliceStridedSlice)decoder/conv2d_transpose_5/Shape:output:07decoder/conv2d_transpose_5/strided_slice/stack:output:09decoder/conv2d_transpose_5/strided_slice/stack_1:output:09decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :љ
 decoder/conv2d_transpose_5/stackPack1decoder/conv2d_transpose_5/strided_slice:output:0+decoder/conv2d_transpose_5/stack/1:output:0+decoder/conv2d_transpose_5/stack/2:output:0+decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
*decoder/conv2d_transpose_5/strided_slice_1StridedSlice)decoder/conv2d_transpose_5/stack:output:09decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskк
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0┬
+decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_5/stack:output:0Bdecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
е
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
"decoder/conv2d_transpose_5/BiasAddBiasAdd4decoder/conv2d_transpose_5/conv2d_transpose:output:09decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ѓ
IdentityIdentity+decoder/conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         Х
NoOpNoOp2^decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp'^decoder/dense_1/BiasAdd/ReadVariableOp&^decoder/dense_1/MatMul/ReadVariableOp(^encoder/conv2d_4/BiasAdd/ReadVariableOp'^encoder/conv2d_4/Conv2D/ReadVariableOp(^encoder/conv2d_5/BiasAdd/ReadVariableOp'^encoder/conv2d_5/Conv2D/ReadVariableOp(^encoder/conv2d_6/BiasAdd/ReadVariableOp'^encoder/conv2d_6/Conv2D/ReadVariableOp(^encoder/conv2d_7/BiasAdd/ReadVariableOp'^encoder/conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2f
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2P
&decoder/dense_1/BiasAdd/ReadVariableOp&decoder/dense_1/BiasAdd/ReadVariableOp2N
%decoder/dense_1/MatMul/ReadVariableOp%decoder/dense_1/MatMul/ReadVariableOp2R
'encoder/conv2d_4/BiasAdd/ReadVariableOp'encoder/conv2d_4/BiasAdd/ReadVariableOp2P
&encoder/conv2d_4/Conv2D/ReadVariableOp&encoder/conv2d_4/Conv2D/ReadVariableOp2R
'encoder/conv2d_5/BiasAdd/ReadVariableOp'encoder/conv2d_5/BiasAdd/ReadVariableOp2P
&encoder/conv2d_5/Conv2D/ReadVariableOp&encoder/conv2d_5/Conv2D/ReadVariableOp2R
'encoder/conv2d_6/BiasAdd/ReadVariableOp'encoder/conv2d_6/BiasAdd/ReadVariableOp2P
&encoder/conv2d_6/Conv2D/ReadVariableOp&encoder/conv2d_6/Conv2D/ReadVariableOp2R
'encoder/conv2d_7/BiasAdd/ReadVariableOp'encoder/conv2d_7/BiasAdd/ReadVariableOp2P
&encoder/conv2d_7/Conv2D/ReadVariableOp&encoder/conv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs"х	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*║
serving_defaultд
C
input_38
serving_default_input_3:0         C
decoder8
StatefulPartitionedCall:0         tensorflow/serving/predict:эЇ
Й
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
ё
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
П
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
 layer_with_weights-3
 layer-5
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_network
ќ
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615"
trackable_list_wrapper
ќ
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
Н
<trace_0
=trace_1
>trace_2
?trace_32Ж
(__inference_model_1_layer_call_fn_365430
(__inference_model_1_layer_call_fn_365737
(__inference_model_1_layer_call_fn_365774
(__inference_model_1_layer_call_fn_365579┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z<trace_0z=trace_1z>trace_2z?trace_3
┴
@trace_0
Atrace_1
Btrace_2
Ctrace_32о
C__inference_model_1_layer_call_and_return_conditional_losses_365886
C__inference_model_1_layer_call_and_return_conditional_losses_365998
C__inference_model_1_layer_call_and_return_conditional_losses_365617
C__inference_model_1_layer_call_and_return_conditional_losses_365655┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z@trace_0zAtrace_1zBtrace_2zCtrace_3
╠B╔
!__inference__wrapped_model_364652input_3"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Д
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_rate
Imomentum_cache'mа(mА)mб*mБ+mц,mЦ-mд.mД/mе0mЕ1mф2mФ3mг4mГ5m«6m»'v░(v▒)v▓*v│+v┤,vх-vХ.vи/vИ0v╣1v║2v╗3v╝4vй5vЙ6v┐"
	optimizer
,
Jserving_default"
signature_map
П
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

'kernel
(bias
 Q_jit_compiled_convolution_op"
_tf_keras_layer
Ц
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
П
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

)kernel
*bias
 ^_jit_compiled_convolution_op"
_tf_keras_layer
Ц
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
П
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

+kernel
,bias
 k_jit_compiled_convolution_op"
_tf_keras_layer
Ц
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
П
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

-kernel
.bias
 x_jit_compiled_convolution_op"
_tf_keras_layer
Ц
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
X
'0
(1
)2
*3
+4
,5
-6
.7"
trackable_list_wrapper
X
'0
(1
)2
*3
+4
,5
-6
.7"
trackable_list_wrapper
 "
trackable_list_wrapper
▒
non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
П
ёtrace_0
Ёtrace_1
єtrace_2
Єtrace_32Ж
(__inference_encoder_layer_call_fn_364800
(__inference_encoder_layer_call_fn_366019
(__inference_encoder_layer_call_fn_366040
(__inference_encoder_layer_call_fn_364931┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zёtrace_0zЁtrace_1zєtrace_2zЄtrace_3
╔
ѕtrace_0
Ѕtrace_1
іtrace_2
Іtrace_32о
C__inference_encoder_layer_call_and_return_conditional_losses_366077
C__inference_encoder_layer_call_and_return_conditional_losses_366114
C__inference_encoder_layer_call_and_return_conditional_losses_364959
C__inference_encoder_layer_call_and_return_conditional_losses_364987┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zѕtrace_0zЅtrace_1zіtrace_2zІtrace_3
"
_tf_keras_input_layer
┴
ї	variables
Їtrainable_variables
јregularization_losses
Ј	keras_api
љ__call__
+Љ&call_and_return_all_conditional_losses

/kernel
0bias"
_tf_keras_layer
Ф
њ	variables
Њtrainable_variables
ћregularization_losses
Ћ	keras_api
ќ__call__
+Ќ&call_and_return_all_conditional_losses"
_tf_keras_layer
С
ў	variables
Ўtrainable_variables
џregularization_losses
Џ	keras_api
ю__call__
+Ю&call_and_return_all_conditional_losses

1kernel
2bias
!ъ_jit_compiled_convolution_op"
_tf_keras_layer
С
Ъ	variables
аtrainable_variables
Аregularization_losses
б	keras_api
Б__call__
+ц&call_and_return_all_conditional_losses

3kernel
4bias
!Ц_jit_compiled_convolution_op"
_tf_keras_layer
С
д	variables
Дtrainable_variables
еregularization_losses
Е	keras_api
ф__call__
+Ф&call_and_return_all_conditional_losses

5kernel
6bias
!г_jit_compiled_convolution_op"
_tf_keras_layer
X
/0
01
12
23
34
45
56
67"
trackable_list_wrapper
X
/0
01
12
23
34
45
56
67"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Гnon_trainable_variables
«layers
»metrics
 ░layer_regularization_losses
▒layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
П
▓trace_0
│trace_1
┤trace_2
хtrace_32Ж
(__inference_decoder_layer_call_fn_365199
(__inference_decoder_layer_call_fn_366135
(__inference_decoder_layer_call_fn_366156
(__inference_decoder_layer_call_fn_365303┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z▓trace_0z│trace_1z┤trace_2zхtrace_3
╔
Хtrace_0
иtrace_1
Иtrace_2
╣trace_32о
C__inference_decoder_layer_call_and_return_conditional_losses_366235
C__inference_decoder_layer_call_and_return_conditional_losses_366314
C__inference_decoder_layer_call_and_return_conditional_losses_365328
C__inference_decoder_layer_call_and_return_conditional_losses_365353┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zХtrace_0zиtrace_1zИtrace_2z╣trace_3
):'2conv2d_4/kernel
:2conv2d_4/bias
):' 2conv2d_5/kernel
: 2conv2d_5/bias
):' @2conv2d_6/kernel
:@2conv2d_6/bias
):'@2conv2d_7/kernel
:2conv2d_7/bias
!:	љ2dense_1/kernel
:љ2dense_1/bias
3:12conv2d_transpose_3/kernel
%:#2conv2d_transpose_3/bias
3:12conv2d_transpose_4/kernel
%:#2conv2d_transpose_4/bias
3:12conv2d_transpose_5/kernel
%:#2conv2d_transpose_5/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
0
║0
╗1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЩBэ
(__inference_model_1_layer_call_fn_365430input_3"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_model_1_layer_call_fn_365737inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_model_1_layer_call_fn_365774inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
(__inference_model_1_layer_call_fn_365579input_3"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_model_1_layer_call_and_return_conditional_losses_365886inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_model_1_layer_call_and_return_conditional_losses_365998inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
C__inference_model_1_layer_call_and_return_conditional_losses_365617input_3"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
C__inference_model_1_layer_call_and_return_conditional_losses_365655input_3"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
╦B╚
$__inference_signature_wrapper_365700input_3"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╝non_trainable_variables
йlayers
Йmetrics
 ┐layer_regularization_losses
└layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
№
┴trace_02л
)__inference_conv2d_4_layer_call_fn_366323б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┴trace_0
і
┬trace_02в
D__inference_conv2d_4_layer_call_and_return_conditional_losses_366334б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┬trace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
├non_trainable_variables
─layers
┼metrics
 кlayer_regularization_losses
Кlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
Ш
╚trace_02О
0__inference_max_pooling2d_3_layer_call_fn_366339б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╚trace_0
Љ
╔trace_02Ы
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_366344б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╔trace_0
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╩non_trainable_variables
╦layers
╠metrics
 ═layer_regularization_losses
╬layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
№
¤trace_02л
)__inference_conv2d_5_layer_call_fn_366353б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z¤trace_0
і
лtrace_02в
D__inference_conv2d_5_layer_call_and_return_conditional_losses_366364б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zлtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Лnon_trainable_variables
мlayers
Мmetrics
 нlayer_regularization_losses
Нlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
Ш
оtrace_02О
0__inference_max_pooling2d_4_layer_call_fn_366369б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zоtrace_0
Љ
Оtrace_02Ы
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_366374б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zОtrace_0
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
пnon_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
№
Пtrace_02л
)__inference_conv2d_6_layer_call_fn_366383б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zПtrace_0
і
яtrace_02в
D__inference_conv2d_6_layer_call_and_return_conditional_losses_366394б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zяtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▀non_trainable_variables
Яlayers
рmetrics
 Рlayer_regularization_losses
сlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
Ш
Сtrace_02О
0__inference_max_pooling2d_5_layer_call_fn_366399б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zСtrace_0
Љ
тtrace_02Ы
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_366404б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zтtrace_0
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Тnon_trainable_variables
уlayers
Уmetrics
 жlayer_regularization_losses
Жlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
№
вtrace_02л
)__inference_conv2d_7_layer_call_fn_366413б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zвtrace_0
і
Вtrace_02в
D__inference_conv2d_7_layer_call_and_return_conditional_losses_366424б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zВtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ьnon_trainable_variables
Ьlayers
№metrics
 ­layer_regularization_losses
ыlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
Ђ
Ыtrace_02Р
;__inference_global_average_pooling2d_1_layer_call_fn_366429б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЫtrace_0
ю
зtrace_02§
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_366435б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zзtrace_0
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЩBэ
(__inference_encoder_layer_call_fn_364800input_3"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_encoder_layer_call_fn_366019inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_encoder_layer_call_fn_366040inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
(__inference_encoder_layer_call_fn_364931input_3"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_encoder_layer_call_and_return_conditional_losses_366077inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_encoder_layer_call_and_return_conditional_losses_366114inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
C__inference_encoder_layer_call_and_return_conditional_losses_364959input_3"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
C__inference_encoder_layer_call_and_return_conditional_losses_364987input_3"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
И
Зnon_trainable_variables
шlayers
Шmetrics
 эlayer_regularization_losses
Эlayer_metrics
ї	variables
Їtrainable_variables
јregularization_losses
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
Ь
щtrace_02¤
(__inference_dense_1_layer_call_fn_366444б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zщtrace_0
Ѕ
Щtrace_02Ж
C__inference_dense_1_layer_call_and_return_conditional_losses_366454б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЩtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
чnon_trainable_variables
Чlayers
§metrics
 ■layer_regularization_losses
 layer_metrics
њ	variables
Њtrainable_variables
ћregularization_losses
ќ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
­
ђtrace_02Л
*__inference_reshape_1_layer_call_fn_366459б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zђtrace_0
І
Ђtrace_02В
E__inference_reshape_1_layer_call_and_return_conditional_losses_366473б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЂtrace_0
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѓnon_trainable_variables
Ѓlayers
ёmetrics
 Ёlayer_regularization_losses
єlayer_metrics
ў	variables
Ўtrainable_variables
џregularization_losses
ю__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
щ
Єtrace_02┌
3__inference_conv2d_transpose_3_layer_call_fn_366482б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЄtrace_0
ћ
ѕtrace_02ш
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_366520б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zѕtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѕnon_trainable_variables
іlayers
Іmetrics
 їlayer_regularization_losses
Їlayer_metrics
Ъ	variables
аtrainable_variables
Аregularization_losses
Б__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
щ
јtrace_02┌
3__inference_conv2d_transpose_4_layer_call_fn_366529б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zјtrace_0
ћ
Јtrace_02ш
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_366563б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЈtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
И
љnon_trainable_variables
Љlayers
њmetrics
 Њlayer_regularization_losses
ћlayer_metrics
д	variables
Дtrainable_variables
еregularization_losses
ф__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
щ
Ћtrace_02┌
3__inference_conv2d_transpose_5_layer_call_fn_366572б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЋtrace_0
ћ
ќtrace_02ш
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_366605б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zќtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
J
0
1
2
3
4
 5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЩBэ
(__inference_decoder_layer_call_fn_365199input_4"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_decoder_layer_call_fn_366135inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
(__inference_decoder_layer_call_fn_366156inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
(__inference_decoder_layer_call_fn_365303input_4"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_decoder_layer_call_and_return_conditional_losses_366235inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћBЉ
C__inference_decoder_layer_call_and_return_conditional_losses_366314inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
C__inference_decoder_layer_call_and_return_conditional_losses_365328input_4"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
C__inference_decoder_layer_call_and_return_conditional_losses_365353input_4"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
R
Ќ	variables
ў	keras_api

Ўtotal

џcount"
_tf_keras_metric
c
Џ	variables
ю	keras_api

Юtotal

ъcount
Ъ
_fn_kwargs"
_tf_keras_metric
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
ПB┌
)__inference_conv2d_4_layer_call_fn_366323inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_conv2d_4_layer_call_and_return_conditional_losses_366334inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
СBр
0__inference_max_pooling2d_3_layer_call_fn_366339inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_366344inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ПB┌
)__inference_conv2d_5_layer_call_fn_366353inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_conv2d_5_layer_call_and_return_conditional_losses_366364inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
СBр
0__inference_max_pooling2d_4_layer_call_fn_366369inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_366374inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ПB┌
)__inference_conv2d_6_layer_call_fn_366383inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_conv2d_6_layer_call_and_return_conditional_losses_366394inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
СBр
0__inference_max_pooling2d_5_layer_call_fn_366399inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_366404inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ПB┌
)__inference_conv2d_7_layer_call_fn_366413inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_conv2d_7_layer_call_and_return_conditional_losses_366424inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
№BВ
;__inference_global_average_pooling2d_1_layer_call_fn_366429inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
іBЄ
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_366435inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▄B┘
(__inference_dense_1_layer_call_fn_366444inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_dense_1_layer_call_and_return_conditional_losses_366454inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
яB█
*__inference_reshape_1_layer_call_fn_366459inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_reshape_1_layer_call_and_return_conditional_losses_366473inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
уBС
3__inference_conv2d_transpose_3_layer_call_fn_366482inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_366520inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
уBС
3__inference_conv2d_transpose_4_layer_call_fn_366529inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_366563inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
уBС
3__inference_conv2d_transpose_5_layer_call_fn_366572inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_366605inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
Ў0
џ1"
trackable_list_wrapper
.
Ќ	variables"
_generic_user_object
:  (2total
:  (2count
0
Ю0
ъ1"
trackable_list_wrapper
.
Џ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:-2Nadam/conv2d_4/kernel/m
!:2Nadam/conv2d_4/bias/m
/:- 2Nadam/conv2d_5/kernel/m
!: 2Nadam/conv2d_5/bias/m
/:- @2Nadam/conv2d_6/kernel/m
!:@2Nadam/conv2d_6/bias/m
/:-@2Nadam/conv2d_7/kernel/m
!:2Nadam/conv2d_7/bias/m
':%	љ2Nadam/dense_1/kernel/m
!:љ2Nadam/dense_1/bias/m
9:72!Nadam/conv2d_transpose_3/kernel/m
+:)2Nadam/conv2d_transpose_3/bias/m
9:72!Nadam/conv2d_transpose_4/kernel/m
+:)2Nadam/conv2d_transpose_4/bias/m
9:72!Nadam/conv2d_transpose_5/kernel/m
+:)2Nadam/conv2d_transpose_5/bias/m
/:-2Nadam/conv2d_4/kernel/v
!:2Nadam/conv2d_4/bias/v
/:- 2Nadam/conv2d_5/kernel/v
!: 2Nadam/conv2d_5/bias/v
/:- @2Nadam/conv2d_6/kernel/v
!:@2Nadam/conv2d_6/bias/v
/:-@2Nadam/conv2d_7/kernel/v
!:2Nadam/conv2d_7/bias/v
':%	љ2Nadam/dense_1/kernel/v
!:љ2Nadam/dense_1/bias/v
9:72!Nadam/conv2d_transpose_3/kernel/v
+:)2Nadam/conv2d_transpose_3/bias/v
9:72!Nadam/conv2d_transpose_4/kernel/v
+:)2Nadam/conv2d_transpose_4/bias/v
9:72!Nadam/conv2d_transpose_5/kernel/v
+:)2Nadam/conv2d_transpose_5/bias/vГ
!__inference__wrapped_model_364652Є'()*+,-./01234568б5
.б+
)і&
input_3         
ф "9ф6
4
decoder)і&
decoder         ┤
D__inference_conv2d_4_layer_call_and_return_conditional_losses_366334l'(7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ ї
)__inference_conv2d_4_layer_call_fn_366323_'(7б4
-б*
(і%
inputs         
ф " і         ┤
D__inference_conv2d_5_layer_call_and_return_conditional_losses_366364l)*7б4
-б*
(і%
inputs         
ф "-б*
#і 
0          
џ ї
)__inference_conv2d_5_layer_call_fn_366353_)*7б4
-б*
(і%
inputs         
ф " і          ┤
D__inference_conv2d_6_layer_call_and_return_conditional_losses_366394l+,7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         @
џ ї
)__inference_conv2d_6_layer_call_fn_366383_+,7б4
-б*
(і%
inputs          
ф " і         @┤
D__inference_conv2d_7_layer_call_and_return_conditional_losses_366424l-.7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         
џ ї
)__inference_conv2d_7_layer_call_fn_366413_-.7б4
-б*
(і%
inputs         @
ф " і         с
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_366520љ12IбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ ╗
3__inference_conv2d_transpose_3_layer_call_fn_366482Ѓ12IбF
?б<
:і7
inputs+                           
ф "2і/+                           с
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_366563љ34IбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ ╗
3__inference_conv2d_transpose_4_layer_call_fn_366529Ѓ34IбF
?б<
:і7
inputs+                           
ф "2і/+                           с
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_366605љ56IбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ ╗
3__inference_conv2d_transpose_5_layer_call_fn_366572Ѓ56IбF
?б<
:і7
inputs+                           
ф "2і/+                           ║
C__inference_decoder_layer_call_and_return_conditional_losses_365328s/01234568б5
.б+
!і
input_4         
p 

 
ф "-б*
#і 
0         
џ ║
C__inference_decoder_layer_call_and_return_conditional_losses_365353s/01234568б5
.б+
!і
input_4         
p

 
ф "-б*
#і 
0         
џ ╣
C__inference_decoder_layer_call_and_return_conditional_losses_366235r/01234567б4
-б*
 і
inputs         
p 

 
ф "-б*
#і 
0         
џ ╣
C__inference_decoder_layer_call_and_return_conditional_losses_366314r/01234567б4
-б*
 і
inputs         
p

 
ф "-б*
#і 
0         
џ њ
(__inference_decoder_layer_call_fn_365199f/01234568б5
.б+
!і
input_4         
p 

 
ф " і         њ
(__inference_decoder_layer_call_fn_365303f/01234568б5
.б+
!і
input_4         
p

 
ф " і         Љ
(__inference_decoder_layer_call_fn_366135e/01234567б4
-б*
 і
inputs         
p 

 
ф " і         Љ
(__inference_decoder_layer_call_fn_366156e/01234567б4
-б*
 і
inputs         
p

 
ф " і         ц
C__inference_dense_1_layer_call_and_return_conditional_losses_366454]/0/б,
%б"
 і
inputs         
ф "&б#
і
0         љ
џ |
(__inference_dense_1_layer_call_fn_366444P/0/б,
%б"
 і
inputs         
ф "і         љ║
C__inference_encoder_layer_call_and_return_conditional_losses_364959s'()*+,-.@б=
6б3
)і&
input_3         
p 

 
ф "%б"
і
0         
џ ║
C__inference_encoder_layer_call_and_return_conditional_losses_364987s'()*+,-.@б=
6б3
)і&
input_3         
p

 
ф "%б"
і
0         
џ ╣
C__inference_encoder_layer_call_and_return_conditional_losses_366077r'()*+,-.?б<
5б2
(і%
inputs         
p 

 
ф "%б"
і
0         
џ ╣
C__inference_encoder_layer_call_and_return_conditional_losses_366114r'()*+,-.?б<
5б2
(і%
inputs         
p

 
ф "%б"
і
0         
џ њ
(__inference_encoder_layer_call_fn_364800f'()*+,-.@б=
6б3
)і&
input_3         
p 

 
ф "і         њ
(__inference_encoder_layer_call_fn_364931f'()*+,-.@б=
6б3
)і&
input_3         
p

 
ф "і         Љ
(__inference_encoder_layer_call_fn_366019e'()*+,-.?б<
5б2
(і%
inputs         
p 

 
ф "і         Љ
(__inference_encoder_layer_call_fn_366040e'()*+,-.?б<
5б2
(і%
inputs         
p

 
ф "і         ▀
V__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_366435ёRбO
HбE
Cі@
inputs4                                    
ф ".б+
$і!
0                  
џ Х
;__inference_global_average_pooling2d_1_layer_call_fn_366429wRбO
HбE
Cі@
inputs4                                    
ф "!і                  Ь
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_366344ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_3_layer_call_fn_366339ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ь
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_366374ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_4_layer_call_fn_366369ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ь
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_366404ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_5_layer_call_fn_366399ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ╦
C__inference_model_1_layer_call_and_return_conditional_losses_365617Ѓ'()*+,-./0123456@б=
6б3
)і&
input_3         
p 

 
ф "-б*
#і 
0         
џ ╦
C__inference_model_1_layer_call_and_return_conditional_losses_365655Ѓ'()*+,-./0123456@б=
6б3
)і&
input_3         
p

 
ф "-б*
#і 
0         
џ ╩
C__inference_model_1_layer_call_and_return_conditional_losses_365886ѓ'()*+,-./0123456?б<
5б2
(і%
inputs         
p 

 
ф "-б*
#і 
0         
џ ╩
C__inference_model_1_layer_call_and_return_conditional_losses_365998ѓ'()*+,-./0123456?б<
5б2
(і%
inputs         
p

 
ф "-б*
#і 
0         
џ б
(__inference_model_1_layer_call_fn_365430v'()*+,-./0123456@б=
6б3
)і&
input_3         
p 

 
ф " і         б
(__inference_model_1_layer_call_fn_365579v'()*+,-./0123456@б=
6б3
)і&
input_3         
p

 
ф " і         А
(__inference_model_1_layer_call_fn_365737u'()*+,-./0123456?б<
5б2
(і%
inputs         
p 

 
ф " і         А
(__inference_model_1_layer_call_fn_365774u'()*+,-./0123456?б<
5б2
(і%
inputs         
p

 
ф " і         ф
E__inference_reshape_1_layer_call_and_return_conditional_losses_366473a0б-
&б#
!і
inputs         љ
ф "-б*
#і 
0         
џ ѓ
*__inference_reshape_1_layer_call_fn_366459T0б-
&б#
!і
inputs         љ
ф " і         ╗
$__inference_signature_wrapper_365700њ'()*+,-./0123456Cб@
б 
9ф6
4
input_3)і&
input_3         "9ф6
4
decoder)і&
decoder         