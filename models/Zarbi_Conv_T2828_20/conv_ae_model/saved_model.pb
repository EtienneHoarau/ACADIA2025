ЅЯ
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
 ѕ"serve*2.10.02unknown8ии
ќ
Nadam/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/conv2d_transpose_2/bias/v
Ј
3Nadam/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose_2/bias/v*
_output_shapes
:*
dtype0
д
!Nadam/conv2d_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Nadam/conv2d_transpose_2/kernel/v
Ъ
5Nadam/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/conv2d_transpose_2/kernel/v*&
_output_shapes
:*
dtype0
ќ
Nadam/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/conv2d_transpose_1/bias/v
Ј
3Nadam/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose_1/bias/v*
_output_shapes
:*
dtype0
д
!Nadam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Nadam/conv2d_transpose_1/kernel/v
Ъ
5Nadam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp!Nadam/conv2d_transpose_1/kernel/v*&
_output_shapes
:*
dtype0
њ
Nadam/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameNadam/conv2d_transpose/bias/v
І
1Nadam/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose/bias/v*
_output_shapes
:*
dtype0
б
Nadam/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/conv2d_transpose/kernel/v
Џ
3Nadam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose/kernel/v*&
_output_shapes
:*
dtype0
}
Nadam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ*#
shared_nameNadam/dense/bias/v
v
&Nadam/dense/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense/bias/v*
_output_shapes	
:љ*
dtype0
Ё
Nadam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љ*%
shared_nameNadam/dense/kernel/v
~
(Nadam/dense/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/v*
_output_shapes
:	љ*
dtype0
ѓ
Nadam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/conv2d_3/bias/v
{
)Nadam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_3/bias/v*
_output_shapes
:*
dtype0
њ
Nadam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameNadam/conv2d_3/kernel/v
І
+Nadam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_3/kernel/v*&
_output_shapes
:@*
dtype0
ѓ
Nadam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameNadam/conv2d_2/bias/v
{
)Nadam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
њ
Nadam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameNadam/conv2d_2/kernel/v
І
+Nadam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
ѓ
Nadam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d_1/bias/v
{
)Nadam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_1/bias/v*
_output_shapes
: *
dtype0
њ
Nadam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameNadam/conv2d_1/kernel/v
І
+Nadam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_1/kernel/v*&
_output_shapes
: *
dtype0
~
Nadam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameNadam/conv2d/bias/v
w
'Nadam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d/bias/v*
_output_shapes
:*
dtype0
ј
Nadam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/conv2d/kernel/v
Є
)Nadam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
ќ
Nadam/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/conv2d_transpose_2/bias/m
Ј
3Nadam/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose_2/bias/m*
_output_shapes
:*
dtype0
д
!Nadam/conv2d_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Nadam/conv2d_transpose_2/kernel/m
Ъ
5Nadam/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/conv2d_transpose_2/kernel/m*&
_output_shapes
:*
dtype0
ќ
Nadam/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/conv2d_transpose_1/bias/m
Ј
3Nadam/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose_1/bias/m*
_output_shapes
:*
dtype0
д
!Nadam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Nadam/conv2d_transpose_1/kernel/m
Ъ
5Nadam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp!Nadam/conv2d_transpose_1/kernel/m*&
_output_shapes
:*
dtype0
њ
Nadam/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameNadam/conv2d_transpose/bias/m
І
1Nadam/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose/bias/m*
_output_shapes
:*
dtype0
б
Nadam/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Nadam/conv2d_transpose/kernel/m
Џ
3Nadam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_transpose/kernel/m*&
_output_shapes
:*
dtype0
}
Nadam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ*#
shared_nameNadam/dense/bias/m
v
&Nadam/dense/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense/bias/m*
_output_shapes	
:љ*
dtype0
Ё
Nadam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љ*%
shared_nameNadam/dense/kernel/m
~
(Nadam/dense/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/m*
_output_shapes
:	љ*
dtype0
ѓ
Nadam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/conv2d_3/bias/m
{
)Nadam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_3/bias/m*
_output_shapes
:*
dtype0
њ
Nadam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameNadam/conv2d_3/kernel/m
І
+Nadam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_3/kernel/m*&
_output_shapes
:@*
dtype0
ѓ
Nadam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameNadam/conv2d_2/bias/m
{
)Nadam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
њ
Nadam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameNadam/conv2d_2/kernel/m
І
+Nadam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
ѓ
Nadam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d_1/bias/m
{
)Nadam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_1/bias/m*
_output_shapes
: *
dtype0
њ
Nadam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameNadam/conv2d_1/kernel/m
І
+Nadam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_1/kernel/m*&
_output_shapes
: *
dtype0
~
Nadam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameNadam/conv2d/bias/m
w
'Nadam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d/bias/m*
_output_shapes
:*
dtype0
ј
Nadam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/conv2d/kernel/m
Є
)Nadam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d/kernel/m*&
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
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
:*
dtype0
ќ
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_2/kernel
Ј
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
:*
dtype0
є
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:*
dtype0
ќ
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_1/kernel
Ј
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:*
dtype0
ѓ
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:*
dtype0
њ
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose/kernel
І
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
:*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:љ*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љ*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	љ*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
ѓ
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
ѓ
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
ѓ
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
і
serving_default_input_1Placeholder*/
_output_shapes
:         *
dtype0*$
shape:         
ѕ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/bias*
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
$__inference_signature_wrapper_178996

NoOpNoOp
ЭЁ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*▓Ё
valueДЁBБЁ BЏЁ
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
MG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv2d_transpose/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_2/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_2/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
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
qk
VARIABLE_VALUENadam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUENadam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUENadam/dense/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUENadam/dense/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/conv2d_transpose/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUENadam/conv2d_transpose/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Nadam/conv2d_transpose_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/conv2d_transpose_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Nadam/conv2d_transpose_2/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/conv2d_transpose_2/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUENadam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUENadam/conv2d_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUENadam/conv2d_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUENadam/dense/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUENadam/dense/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/conv2d_transpose/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUENadam/conv2d_transpose/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Nadam/conv2d_transpose_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/conv2d_transpose_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Nadam/conv2d_transpose_2/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUENadam/conv2d_transpose_2/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
у
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Nadam/conv2d/kernel/m/Read/ReadVariableOp'Nadam/conv2d/bias/m/Read/ReadVariableOp+Nadam/conv2d_1/kernel/m/Read/ReadVariableOp)Nadam/conv2d_1/bias/m/Read/ReadVariableOp+Nadam/conv2d_2/kernel/m/Read/ReadVariableOp)Nadam/conv2d_2/bias/m/Read/ReadVariableOp+Nadam/conv2d_3/kernel/m/Read/ReadVariableOp)Nadam/conv2d_3/bias/m/Read/ReadVariableOp(Nadam/dense/kernel/m/Read/ReadVariableOp&Nadam/dense/bias/m/Read/ReadVariableOp3Nadam/conv2d_transpose/kernel/m/Read/ReadVariableOp1Nadam/conv2d_transpose/bias/m/Read/ReadVariableOp5Nadam/conv2d_transpose_1/kernel/m/Read/ReadVariableOp3Nadam/conv2d_transpose_1/bias/m/Read/ReadVariableOp5Nadam/conv2d_transpose_2/kernel/m/Read/ReadVariableOp3Nadam/conv2d_transpose_2/bias/m/Read/ReadVariableOp)Nadam/conv2d/kernel/v/Read/ReadVariableOp'Nadam/conv2d/bias/v/Read/ReadVariableOp+Nadam/conv2d_1/kernel/v/Read/ReadVariableOp)Nadam/conv2d_1/bias/v/Read/ReadVariableOp+Nadam/conv2d_2/kernel/v/Read/ReadVariableOp)Nadam/conv2d_2/bias/v/Read/ReadVariableOp+Nadam/conv2d_3/kernel/v/Read/ReadVariableOp)Nadam/conv2d_3/bias/v/Read/ReadVariableOp(Nadam/dense/kernel/v/Read/ReadVariableOp&Nadam/dense/bias/v/Read/ReadVariableOp3Nadam/conv2d_transpose/kernel/v/Read/ReadVariableOp1Nadam/conv2d_transpose/bias/v/Read/ReadVariableOp5Nadam/conv2d_transpose_1/kernel/v/Read/ReadVariableOp3Nadam/conv2d_transpose_1/bias/v/Read/ReadVariableOp5Nadam/conv2d_transpose_2/kernel/v/Read/ReadVariableOp3Nadam/conv2d_transpose_2/bias/v/Read/ReadVariableOpConst*G
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
__inference__traced_save_180098
┌
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotal_1count_1totalcountNadam/conv2d/kernel/mNadam/conv2d/bias/mNadam/conv2d_1/kernel/mNadam/conv2d_1/bias/mNadam/conv2d_2/kernel/mNadam/conv2d_2/bias/mNadam/conv2d_3/kernel/mNadam/conv2d_3/bias/mNadam/dense/kernel/mNadam/dense/bias/mNadam/conv2d_transpose/kernel/mNadam/conv2d_transpose/bias/m!Nadam/conv2d_transpose_1/kernel/mNadam/conv2d_transpose_1/bias/m!Nadam/conv2d_transpose_2/kernel/mNadam/conv2d_transpose_2/bias/mNadam/conv2d/kernel/vNadam/conv2d/bias/vNadam/conv2d_1/kernel/vNadam/conv2d_1/bias/vNadam/conv2d_2/kernel/vNadam/conv2d_2/bias/vNadam/conv2d_3/kernel/vNadam/conv2d_3/bias/vNadam/dense/kernel/vNadam/dense/bias/vNadam/conv2d_transpose/kernel/vNadam/conv2d_transpose/bias/v!Nadam/conv2d_transpose_1/kernel/vNadam/conv2d_transpose_1/bias/v!Nadam/conv2d_transpose_2/kernel/vNadam/conv2d_transpose_2/bias/v*F
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
"__inference__traced_restore_180282Яљ
х
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_177994

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
Ж
ъ
)__inference_conv2d_1_layer_call_fn_179649

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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_178033w
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
(__inference_encoder_layer_call_fn_179336

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
C__inference_encoder_layer_call_and_return_conditional_losses_178187o
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
Љ
█
C__inference_decoder_layer_call_and_return_conditional_losses_178476

inputs
dense_178439:	љ
dense_178441:	љ1
conv2d_transpose_178460:%
conv2d_transpose_178462:3
conv2d_transpose_1_178465:'
conv2d_transpose_1_178467:3
conv2d_transpose_2_178470:'
conv2d_transpose_2_178472:
identityѕб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallбdense/StatefulPartitionedCallт
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_178439dense_178441*
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
GPU 2J 8ѓ *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_178438я
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_178458▓
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_178460conv2d_transpose_178462*
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
GPU 2J 8ѓ *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_178325╦
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_178465conv2d_transpose_1_178467*
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_178370═
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_178470conv2d_transpose_2_178472*
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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_178414і
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         в
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╩c
З
C__inference_decoder_layer_call_and_return_conditional_losses_179531

inputs7
$dense_matmul_readvariableop_resource:	љ4
%dense_biasadd_readvariableop_resource:	љS
9conv2d_transpose_conv2d_transpose_readvariableop_resource:>
0conv2d_transpose_biasadd_readvariableop_resource:U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:U
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_2_biasadd_readvariableop_resource:
identityѕб'conv2d_transpose/BiasAdd/ReadVariableOpб0conv2d_transpose/conv2d_transpose/ReadVariableOpб)conv2d_transpose_1/BiasAdd/ReadVariableOpб2conv2d_transpose_1/conv2d_transpose/ReadVariableOpб)conv2d_transpose_2/BiasAdd/ReadVariableOpб2conv2d_transpose_2/conv2d_transpose/ReadVariableOpбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpЂ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ѕ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љS
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Л
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ї
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         ^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :я
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask▓
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0љ
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
ћ
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0║
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:         k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0а
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:         m
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0б
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
IdentityIdentity#conv2d_transpose_2/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         б
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
В
╠
&__inference_model_layer_call_fn_179070

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
identityѕбStatefulPartitionedCallЎ
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
GPU 2J 8ѓ *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_178803w
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
ћ
▄
C__inference_decoder_layer_call_and_return_conditional_losses_178649
input_2
dense_178627:	љ
dense_178629:	љ1
conv2d_transpose_178633:%
conv2d_transpose_178635:3
conv2d_transpose_1_178638:'
conv2d_transpose_1_178640:3
conv2d_transpose_2_178643:'
conv2d_transpose_2_178645:
identityѕб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallбdense/StatefulPartitionedCallТ
dense/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_178627dense_178629*
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
GPU 2J 8ѓ *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_178438я
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_178458▓
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_178633conv2d_transpose_178635*
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
GPU 2J 8ѓ *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_178325╦
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_178638conv2d_transpose_1_178640*
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_178370═
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_178643conv2d_transpose_2_178645*
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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_178414і
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         в
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
─!
Џ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_178370

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
Ћ
U
9__inference_global_average_pooling2d_layer_call_fn_179725

inputs
identity╚
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
GPU 2J 8ѓ *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_177994i
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
Ѓ
§
D__inference_conv2d_3_layer_call_and_return_conditional_losses_178069

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
╦	
З
A__inference_dense_layer_call_and_return_conditional_losses_178438

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
Ж
ъ
)__inference_conv2d_3_layer_call_fn_179709

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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_178069w
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
Т
ю
'__inference_conv2d_layer_call_fn_179619

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall▀
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
GPU 2J 8ѓ *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_178015w
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
■"
Љ
C__inference_encoder_layer_call_and_return_conditional_losses_178255
input_1'
conv2d_178230:
conv2d_178232:)
conv2d_1_178236: 
conv2d_1_178238: )
conv2d_2_178242: @
conv2d_2_178244:@)
conv2d_3_178248:@
conv2d_3_178250:
identityѕбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallы
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_178230conv2d_178232*
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
GPU 2J 8ѓ *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_178015в
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_177957ў
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_178236conv2d_1_178238*
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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_178033ы
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_177969џ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_178242conv2d_2_178244*
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_178051ы
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_177981џ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_178248conv2d_3_178250*
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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_178069ч
(global_average_pooling2d/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_177994ђ
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         л
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
юЪ
Р
A__inference_model_layer_call_and_return_conditional_losses_179294

inputsG
-encoder_conv2d_conv2d_readvariableop_resource:<
.encoder_conv2d_biasadd_readvariableop_resource:I
/encoder_conv2d_1_conv2d_readvariableop_resource: >
0encoder_conv2d_1_biasadd_readvariableop_resource: I
/encoder_conv2d_2_conv2d_readvariableop_resource: @>
0encoder_conv2d_2_biasadd_readvariableop_resource:@I
/encoder_conv2d_3_conv2d_readvariableop_resource:@>
0encoder_conv2d_3_biasadd_readvariableop_resource:?
,decoder_dense_matmul_readvariableop_resource:	љ<
-decoder_dense_biasadd_readvariableop_resource:	љ[
Adecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource:F
8decoder_conv2d_transpose_biasadd_readvariableop_resource:]
Cdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_1_biasadd_readvariableop_resource:]
Cdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_2_biasadd_readvariableop_resource:
identityѕб/decoder/conv2d_transpose/BiasAdd/ReadVariableOpб8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpб1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpб:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpб1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpб:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpб$decoder/dense/BiasAdd/ReadVariableOpб#decoder/dense/MatMul/ReadVariableOpб%encoder/conv2d/BiasAdd/ReadVariableOpб$encoder/conv2d/Conv2D/ReadVariableOpб'encoder/conv2d_1/BiasAdd/ReadVariableOpб&encoder/conv2d_1/Conv2D/ReadVariableOpб'encoder/conv2d_2/BiasAdd/ReadVariableOpб&encoder/conv2d_2/Conv2D/ReadVariableOpб'encoder/conv2d_3/BiasAdd/ReadVariableOpб&encoder/conv2d_3/Conv2D/ReadVariableOpџ
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0и
encoder/conv2d/Conv2DConv2Dinputs,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
љ
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ф
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         v
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         И
encoder/max_pooling2d/MaxPoolMaxPool!encoder/conv2d/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
ъ
&encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0█
encoder/conv2d_1/Conv2DConv2D&encoder/max_pooling2d/MaxPool:output:0.encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ћ
'encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0░
encoder/conv2d_1/BiasAddBiasAdd encoder/conv2d_1/Conv2D:output:0/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          z
encoder/conv2d_1/ReluRelu!encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
encoder/max_pooling2d_1/MaxPoolMaxPool#encoder/conv2d_1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
ъ
&encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0П
encoder/conv2d_2/Conv2DConv2D(encoder/max_pooling2d_1/MaxPool:output:0.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ћ
'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0░
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @z
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @╝
encoder/max_pooling2d_2/MaxPoolMaxPool#encoder/conv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ъ
&encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0П
encoder/conv2d_3/Conv2DConv2D(encoder/max_pooling2d_2/MaxPool:output:0.encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ћ
'encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0░
encoder/conv2d_3/BiasAddBiasAdd encoder/conv2d_3/Conv2D:output:0/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
encoder/conv2d_3/ReluRelu!encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:         ѕ
7encoder/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      к
%encoder/global_average_pooling2d/MeanMean#encoder/conv2d_3/Relu:activations:0@encoder/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Љ
#decoder/dense/MatMul/ReadVariableOpReadVariableOp,decoder_dense_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0«
decoder/dense/MatMulMatMul.encoder/global_average_pooling2d/Mean:output:0+decoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЈ
$decoder/dense/BiasAdd/ReadVariableOpReadVariableOp-decoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0А
decoder/dense/BiasAddBiasAdddecoder/dense/MatMul:product:0,decoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љc
decoder/reshape/ShapeShapedecoder/dense/BiasAdd:output:0*
T0*
_output_shapes
:m
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :a
decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :щ
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0(decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ц
decoder/reshape/ReshapeReshapedecoder/dense/BiasAdd:output:0&decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         n
decoder/conv2d_transpose/ShapeShape decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:v
,decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
&decoder/conv2d_transpose/strided_sliceStridedSlice'decoder/conv2d_transpose/Shape:output:05decoder/conv2d_transpose/strided_slice/stack:output:07decoder/conv2d_transpose/strided_slice/stack_1:output:07decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :є
decoder/conv2d_transpose/stackPack/decoder/conv2d_transpose/strided_slice:output:0)decoder/conv2d_transpose/stack/1:output:0)decoder/conv2d_transpose/stack/2:output:0)decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:x
.decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
(decoder/conv2d_transpose/strided_slice_1StridedSlice'decoder/conv2d_transpose/stack:output:07decoder/conv2d_transpose/strided_slice_1/stack:output:09decoder/conv2d_transpose/strided_slice_1/stack_1:output:09decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┬
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0░
)decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput'decoder/conv2d_transpose/stack:output:0@decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0 decoder/reshape/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
ц
/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0м
 decoder/conv2d_transpose/BiasAddBiasAdd2decoder/conv2d_transpose/conv2d_transpose:output:07decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         і
decoder/conv2d_transpose/ReluRelu)decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:         {
 decoder/conv2d_transpose_1/ShapeShape+decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
(decoder/conv2d_transpose_1/strided_sliceStridedSlice)decoder/conv2d_transpose_1/Shape:output:07decoder/conv2d_transpose_1/strided_slice/stack:output:09decoder/conv2d_transpose_1/strided_slice/stack_1:output:09decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :љ
 decoder/conv2d_transpose_1/stackPack1decoder/conv2d_transpose_1/strided_slice:output:0+decoder/conv2d_transpose_1/stack/1:output:0+decoder/conv2d_transpose_1/stack/2:output:0+decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
*decoder/conv2d_transpose_1/strided_slice_1StridedSlice)decoder/conv2d_transpose_1/stack:output:09decoder/conv2d_transpose_1/strided_slice_1/stack:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskк
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0└
+decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_1/stack:output:0Bdecoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0+decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
е
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
"decoder/conv2d_transpose_1/BiasAddBiasAdd4decoder/conv2d_transpose_1/conv2d_transpose:output:09decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ј
decoder/conv2d_transpose_1/ReluRelu+decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:         }
 decoder/conv2d_transpose_2/ShapeShape-decoder/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
(decoder/conv2d_transpose_2/strided_sliceStridedSlice)decoder/conv2d_transpose_2/Shape:output:07decoder/conv2d_transpose_2/strided_slice/stack:output:09decoder/conv2d_transpose_2/strided_slice/stack_1:output:09decoder/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :љ
 decoder/conv2d_transpose_2/stackPack1decoder/conv2d_transpose_2/strided_slice:output:0+decoder/conv2d_transpose_2/stack/1:output:0+decoder/conv2d_transpose_2/stack/2:output:0+decoder/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
*decoder/conv2d_transpose_2/strided_slice_1StridedSlice)decoder/conv2d_transpose_2/stack:output:09decoder/conv2d_transpose_2/strided_slice_1/stack:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskк
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0┬
+decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_2/stack:output:0Bdecoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
е
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
"decoder/conv2d_transpose_2/BiasAddBiasAdd4decoder/conv2d_transpose_2/conv2d_transpose:output:09decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ѓ
IdentityIdentity+decoder/conv2d_transpose_2/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         ф
NoOpNoOp0^decoder/conv2d_transpose/BiasAdd/ReadVariableOp9^decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp%^decoder/dense/BiasAdd/ReadVariableOp$^decoder/dense/MatMul/ReadVariableOp&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp(^encoder/conv2d_3/BiasAdd/ReadVariableOp'^encoder/conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2b
/decoder/conv2d_transpose/BiasAdd/ReadVariableOp/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2t
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2L
$decoder/dense/BiasAdd/ReadVariableOp$decoder/dense/BiasAdd/ReadVariableOp2J
#decoder/dense/MatMul/ReadVariableOp#decoder/dense/MatMul/ReadVariableOp2N
%encoder/conv2d/BiasAdd/ReadVariableOp%encoder/conv2d/BiasAdd/ReadVariableOp2L
$encoder/conv2d/Conv2D/ReadVariableOp$encoder/conv2d/Conv2D/ReadVariableOp2R
'encoder/conv2d_1/BiasAdd/ReadVariableOp'encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&encoder/conv2d_1/Conv2D/ReadVariableOp&encoder/conv2d_1/Conv2D/ReadVariableOp2R
'encoder/conv2d_2/BiasAdd/ReadVariableOp'encoder/conv2d_2/BiasAdd/ReadVariableOp2P
&encoder/conv2d_2/Conv2D/ReadVariableOp&encoder/conv2d_2/Conv2D/ReadVariableOp2R
'encoder/conv2d_3/BiasAdd/ReadVariableOp'encoder/conv2d_3/BiasAdd/ReadVariableOp2P
&encoder/conv2d_3/Conv2D/ReadVariableOp&encoder/conv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
─!
Џ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_179859

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
Ж
ъ
)__inference_conv2d_2_layer_call_fn_179679

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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_178051w
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
╔
я
A__inference_model_layer_call_and_return_conditional_losses_178803

inputs(
encoder_178768:
encoder_178770:(
encoder_178772: 
encoder_178774: (
encoder_178776: @
encoder_178778:@(
encoder_178780:@
encoder_178782:!
decoder_178785:	љ
decoder_178787:	љ(
decoder_178789:
decoder_178791:(
decoder_178793:
decoder_178795:(
decoder_178797:
decoder_178799:
identityѕбdecoder/StatefulPartitionedCallбencoder/StatefulPartitionedCallп
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_178768encoder_178770encoder_178772encoder_178774encoder_178776encoder_178778encoder_178780encoder_178782*
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
C__inference_encoder_layer_call_and_return_conditional_losses_178187ѓ
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_178785decoder_178787decoder_178789decoder_178791decoder_178793decoder_178795decoder_178797decoder_178799*
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
C__inference_decoder_layer_call_and_return_conditional_losses_178559
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
■"
Љ
C__inference_encoder_layer_call_and_return_conditional_losses_178283
input_1'
conv2d_178258:
conv2d_178260:)
conv2d_1_178264: 
conv2d_1_178266: )
conv2d_2_178270: @
conv2d_2_178272:@)
conv2d_3_178276:@
conv2d_3_178278:
identityѕбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallы
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_178258conv2d_178260*
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
GPU 2J 8ѓ *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_178015в
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_177957ў
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_178264conv2d_1_178266*
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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_178033ы
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_177969џ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_178270conv2d_2_178272*
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_178051ы
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_177981џ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_178276conv2d_3_178278*
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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_178069ч
(global_average_pooling2d/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_177994ђ
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         л
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
╩c
З
C__inference_decoder_layer_call_and_return_conditional_losses_179610

inputs7
$dense_matmul_readvariableop_resource:	љ4
%dense_biasadd_readvariableop_resource:	љS
9conv2d_transpose_conv2d_transpose_readvariableop_resource:>
0conv2d_transpose_biasadd_readvariableop_resource:U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:U
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_2_biasadd_readvariableop_resource:
identityѕб'conv2d_transpose/BiasAdd/ReadVariableOpб0conv2d_transpose/conv2d_transpose/ReadVariableOpб)conv2d_transpose_1/BiasAdd/ReadVariableOpб2conv2d_transpose_1/conv2d_transpose/ReadVariableOpб)conv2d_transpose_2/BiasAdd/ReadVariableOpб2conv2d_transpose_2/conv2d_transpose/ReadVariableOpбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpЂ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0Ѕ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љS
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Л
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ї
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         ^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :я
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask▓
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0љ
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
ћ
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0║
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:         k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0а
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:         m
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :У
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskХ
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0б
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ў
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0└
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
IdentityIdentity#conv2d_transpose_2/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         б
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
И
L
0__inference_max_pooling2d_2_layer_call_fn_179695

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
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_177981Ѓ
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
Т	
Л
(__inference_decoder_layer_call_fn_179452

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
C__inference_decoder_layer_call_and_return_conditional_losses_178559w
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
ћ
▄
C__inference_decoder_layer_call_and_return_conditional_losses_178624
input_2
dense_178602:	љ
dense_178604:	љ1
conv2d_transpose_178608:%
conv2d_transpose_178610:3
conv2d_transpose_1_178613:'
conv2d_transpose_1_178615:3
conv2d_transpose_2_178618:'
conv2d_transpose_2_178620:
identityѕб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallбdense/StatefulPartitionedCallТ
dense/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_178602dense_178604*
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
GPU 2J 8ѓ *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_178438я
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_178458▓
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_178608conv2d_transpose_178610*
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
GPU 2J 8ѓ *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_178325╦
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_178613conv2d_transpose_1_178615*
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_178370═
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_178618conv2d_transpose_2_178620*
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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_178414і
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         в
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
ж	
м
(__inference_decoder_layer_call_fn_178495
input_2
unknown:	љ
	unknown_0:	љ#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identityѕбStatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
C__inference_decoder_layer_call_and_return_conditional_losses_178476w
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
_user_specified_name	input_2
Т	
Л
(__inference_decoder_layer_call_fn_179431

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
C__inference_decoder_layer_call_and_return_conditional_losses_178476w
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
К
е
3__inference_conv2d_transpose_2_layer_call_fn_179868

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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_178414Ѕ
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
└
Ћ
&__inference_dense_layer_call_fn_179740

inputs
unknown:	љ
	unknown_0:	љ
identityѕбStatefulPartitionedCallО
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
GPU 2J 8ѓ *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_178438p
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
Ђ
ч
B__inference_conv2d_layer_call_and_return_conditional_losses_178015

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
╦	
З
A__inference_dense_layer_call_and_return_conditional_losses_179750

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
Њ
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_177981

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
о 
Џ
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_178414

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
Х-
м
C__inference_encoder_layer_call_and_return_conditional_losses_179410

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@6
(conv2d_3_biasadd_readvariableop_resource:
identityѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpі
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Д
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ђ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0њ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         е
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
ј
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0├
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ё
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:          г
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
ј
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0┼
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ё
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ў
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @г
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ј
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0┼
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ё
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:         ђ
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      «
global_average_pooling2d/MeanMeanconv2d_3/Relu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         u
IdentityIdentity&global_average_pooling2d/Mean:output:0^NoOp*
T0*'
_output_shapes
:         ╬
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
х
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_179731

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
Ѓ
§
D__inference_conv2d_1_layer_call_and_return_conditional_losses_178033

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
ж	
м
(__inference_decoder_layer_call_fn_178599
input_2
unknown:	љ
	unknown_0:	љ#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identityѕбStatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
C__inference_decoder_layer_call_and_return_conditional_losses_178559w
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
_user_specified_name	input_2
Њ
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_179700

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
╠
▀
A__inference_model_layer_call_and_return_conditional_losses_178913
input_1(
encoder_178878:
encoder_178880:(
encoder_178882: 
encoder_178884: (
encoder_178886: @
encoder_178888:@(
encoder_178890:@
encoder_178892:!
decoder_178895:	љ
decoder_178897:	љ(
decoder_178899:
decoder_178901:(
decoder_178903:
decoder_178905:(
decoder_178907:
decoder_178909:
identityѕбdecoder/StatefulPartitionedCallбencoder/StatefulPartitionedCall┘
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_178878encoder_178880encoder_178882encoder_178884encoder_178886encoder_178888encoder_178890encoder_178892*
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
C__inference_encoder_layer_call_and_return_conditional_losses_178077ѓ
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_178895decoder_178897decoder_178899decoder_178901decoder_178903decoder_178905decoder_178907decoder_178909*
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
C__inference_decoder_layer_call_and_return_conditional_losses_178476
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
_user_specified_name	input_1
№	
п
(__inference_encoder_layer_call_fn_178227
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@
	unknown_6:
identityѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
C__inference_encoder_layer_call_and_return_conditional_losses_178187o
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
_user_specified_name	input_1
№
═
&__inference_model_layer_call_fn_178726
input_1!
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
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8ѓ *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_178691w
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
_user_specified_name	input_1
Љ
█
C__inference_decoder_layer_call_and_return_conditional_losses_178559

inputs
dense_178537:	љ
dense_178539:	љ1
conv2d_transpose_178543:%
conv2d_transpose_178545:3
conv2d_transpose_1_178548:'
conv2d_transpose_1_178550:3
conv2d_transpose_2_178553:'
conv2d_transpose_2_178555:
identityѕб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallбdense/StatefulPartitionedCallт
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_178537dense_178539*
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
GPU 2J 8ѓ *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_178438я
reshape/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_178458▓
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_178543conv2d_transpose_178545*
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
GPU 2J 8ѓ *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_178325╦
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_178548conv2d_transpose_1_178550*
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_178370═
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_178553conv2d_transpose_2_178555*
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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_178414і
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         в
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╝г
Ѓ
!__inference__wrapped_model_177948
input_1M
3model_encoder_conv2d_conv2d_readvariableop_resource:B
4model_encoder_conv2d_biasadd_readvariableop_resource:O
5model_encoder_conv2d_1_conv2d_readvariableop_resource: D
6model_encoder_conv2d_1_biasadd_readvariableop_resource: O
5model_encoder_conv2d_2_conv2d_readvariableop_resource: @D
6model_encoder_conv2d_2_biasadd_readvariableop_resource:@O
5model_encoder_conv2d_3_conv2d_readvariableop_resource:@D
6model_encoder_conv2d_3_biasadd_readvariableop_resource:E
2model_decoder_dense_matmul_readvariableop_resource:	љB
3model_decoder_dense_biasadd_readvariableop_resource:	љa
Gmodel_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource:L
>model_decoder_conv2d_transpose_biasadd_readvariableop_resource:c
Imodel_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:N
@model_decoder_conv2d_transpose_1_biasadd_readvariableop_resource:c
Imodel_decoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:N
@model_decoder_conv2d_transpose_2_biasadd_readvariableop_resource:
identityѕб5model/decoder/conv2d_transpose/BiasAdd/ReadVariableOpб>model/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpб7model/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpб@model/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpб7model/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpб@model/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpб*model/decoder/dense/BiasAdd/ReadVariableOpб)model/decoder/dense/MatMul/ReadVariableOpб+model/encoder/conv2d/BiasAdd/ReadVariableOpб*model/encoder/conv2d/Conv2D/ReadVariableOpб-model/encoder/conv2d_1/BiasAdd/ReadVariableOpб,model/encoder/conv2d_1/Conv2D/ReadVariableOpб-model/encoder/conv2d_2/BiasAdd/ReadVariableOpб,model/encoder/conv2d_2/Conv2D/ReadVariableOpб-model/encoder/conv2d_3/BiasAdd/ReadVariableOpб,model/encoder/conv2d_3/Conv2D/ReadVariableOpд
*model/encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp3model_encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0─
model/encoder/conv2d/Conv2DConv2Dinput_12model/encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ю
+model/encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp4model_encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╝
model/encoder/conv2d/BiasAddBiasAdd$model/encoder/conv2d/Conv2D:output:03model/encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ѓ
model/encoder/conv2d/ReluRelu%model/encoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         ─
#model/encoder/max_pooling2d/MaxPoolMaxPool'model/encoder/conv2d/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
ф
,model/encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5model_encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ь
model/encoder/conv2d_1/Conv2DConv2D,model/encoder/max_pooling2d/MaxPool:output:04model/encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
а
-model/encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6model_encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┬
model/encoder/conv2d_1/BiasAddBiasAdd&model/encoder/conv2d_1/Conv2D:output:05model/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          є
model/encoder/conv2d_1/ReluRelu'model/encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:          ╚
%model/encoder/max_pooling2d_1/MaxPoolMaxPool)model/encoder/conv2d_1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
ф
,model/encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5model_encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0№
model/encoder/conv2d_2/Conv2DConv2D.model/encoder/max_pooling2d_1/MaxPool:output:04model/encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
а
-model/encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6model_encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┬
model/encoder/conv2d_2/BiasAddBiasAdd&model/encoder/conv2d_2/Conv2D:output:05model/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @є
model/encoder/conv2d_2/ReluRelu'model/encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @╚
%model/encoder/max_pooling2d_2/MaxPoolMaxPool)model/encoder/conv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ф
,model/encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp5model_encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0№
model/encoder/conv2d_3/Conv2DConv2D.model/encoder/max_pooling2d_2/MaxPool:output:04model/encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
а
-model/encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp6model_encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
model/encoder/conv2d_3/BiasAddBiasAdd&model/encoder/conv2d_3/Conv2D:output:05model/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         є
model/encoder/conv2d_3/ReluRelu'model/encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:         ј
=model/encoder/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      п
+model/encoder/global_average_pooling2d/MeanMean)model/encoder/conv2d_3/Relu:activations:0Fmodel/encoder/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Ю
)model/decoder/dense/MatMul/ReadVariableOpReadVariableOp2model_decoder_dense_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0└
model/decoder/dense/MatMulMatMul4model/encoder/global_average_pooling2d/Mean:output:01model/decoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЏ
*model/decoder/dense/BiasAdd/ReadVariableOpReadVariableOp3model_decoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0│
model/decoder/dense/BiasAddBiasAdd$model/decoder/dense/MatMul:product:02model/decoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љo
model/decoder/reshape/ShapeShape$model/decoder/dense/BiasAdd:output:0*
T0*
_output_shapes
:s
)model/decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+model/decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+model/decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#model/decoder/reshape/strided_sliceStridedSlice$model/decoder/reshape/Shape:output:02model/decoder/reshape/strided_slice/stack:output:04model/decoder/reshape/strided_slice/stack_1:output:04model/decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%model/decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :g
%model/decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :g
%model/decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ќ
#model/decoder/reshape/Reshape/shapePack,model/decoder/reshape/strided_slice:output:0.model/decoder/reshape/Reshape/shape/1:output:0.model/decoder/reshape/Reshape/shape/2:output:0.model/decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Х
model/decoder/reshape/ReshapeReshape$model/decoder/dense/BiasAdd:output:0,model/decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         z
$model/decoder/conv2d_transpose/ShapeShape&model/decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:|
2model/decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4model/decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4model/decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:В
,model/decoder/conv2d_transpose/strided_sliceStridedSlice-model/decoder/conv2d_transpose/Shape:output:0;model/decoder/conv2d_transpose/strided_slice/stack:output:0=model/decoder/conv2d_transpose/strided_slice/stack_1:output:0=model/decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&model/decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :h
&model/decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :h
&model/decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :ц
$model/decoder/conv2d_transpose/stackPack5model/decoder/conv2d_transpose/strided_slice:output:0/model/decoder/conv2d_transpose/stack/1:output:0/model/decoder/conv2d_transpose/stack/2:output:0/model/decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:~
4model/decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ђ
6model/decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ђ
6model/decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
.model/decoder/conv2d_transpose/strided_slice_1StridedSlice-model/decoder/conv2d_transpose/stack:output:0=model/decoder/conv2d_transpose/strided_slice_1/stack:output:0?model/decoder/conv2d_transpose/strided_slice_1/stack_1:output:0?model/decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╬
>model/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpGmodel_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0╚
/model/decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput-model/decoder/conv2d_transpose/stack:output:0Fmodel/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0&model/decoder/reshape/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
░
5model/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp>model_decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
&model/decoder/conv2d_transpose/BiasAddBiasAdd8model/decoder/conv2d_transpose/conv2d_transpose:output:0=model/decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ќ
#model/decoder/conv2d_transpose/ReluRelu/model/decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:         Є
&model/decoder/conv2d_transpose_1/ShapeShape1model/decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:~
4model/decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ђ
6model/decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ђ
6model/decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ш
.model/decoder/conv2d_transpose_1/strided_sliceStridedSlice/model/decoder/conv2d_transpose_1/Shape:output:0=model/decoder/conv2d_transpose_1/strided_slice/stack:output:0?model/decoder/conv2d_transpose_1/strided_slice/stack_1:output:0?model/decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :j
(model/decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :j
(model/decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :«
&model/decoder/conv2d_transpose_1/stackPack7model/decoder/conv2d_transpose_1/strided_slice:output:01model/decoder/conv2d_transpose_1/stack/1:output:01model/decoder/conv2d_transpose_1/stack/2:output:01model/decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:ђ
6model/decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ѓ
8model/decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ѓ
8model/decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0model/decoder/conv2d_transpose_1/strided_slice_1StridedSlice/model/decoder/conv2d_transpose_1/stack:output:0?model/decoder/conv2d_transpose_1/strided_slice_1/stack:output:0Amodel/decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0Amodel/decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskм
@model/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpImodel_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0п
1model/decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput/model/decoder/conv2d_transpose_1/stack:output:0Hmodel/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:01model/decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
┤
7model/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp@model_decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж
(model/decoder/conv2d_transpose_1/BiasAddBiasAdd:model/decoder/conv2d_transpose_1/conv2d_transpose:output:0?model/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         џ
%model/decoder/conv2d_transpose_1/ReluRelu1model/decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:         Ѕ
&model/decoder/conv2d_transpose_2/ShapeShape3model/decoder/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:~
4model/decoder/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ђ
6model/decoder/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ђ
6model/decoder/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ш
.model/decoder/conv2d_transpose_2/strided_sliceStridedSlice/model/decoder/conv2d_transpose_2/Shape:output:0=model/decoder/conv2d_transpose_2/strided_slice/stack:output:0?model/decoder/conv2d_transpose_2/strided_slice/stack_1:output:0?model/decoder/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(model/decoder/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :j
(model/decoder/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :j
(model/decoder/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :«
&model/decoder/conv2d_transpose_2/stackPack7model/decoder/conv2d_transpose_2/strided_slice:output:01model/decoder/conv2d_transpose_2/stack/1:output:01model/decoder/conv2d_transpose_2/stack/2:output:01model/decoder/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:ђ
6model/decoder/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ѓ
8model/decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ѓ
8model/decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
0model/decoder/conv2d_transpose_2/strided_slice_1StridedSlice/model/decoder/conv2d_transpose_2/stack:output:0?model/decoder/conv2d_transpose_2/strided_slice_1/stack:output:0Amodel/decoder/conv2d_transpose_2/strided_slice_1/stack_1:output:0Amodel/decoder/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskм
@model/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpImodel_decoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0┌
1model/decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput/model/decoder/conv2d_transpose_2/stack:output:0Hmodel/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:03model/decoder/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
┤
7model/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp@model_decoder_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж
(model/decoder/conv2d_transpose_2/BiasAddBiasAdd:model/decoder/conv2d_transpose_2/conv2d_transpose:output:0?model/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ѕ
IdentityIdentity1model/decoder/conv2d_transpose_2/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         і
NoOpNoOp6^model/decoder/conv2d_transpose/BiasAdd/ReadVariableOp?^model/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp8^model/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpA^model/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp8^model/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpA^model/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp+^model/decoder/dense/BiasAdd/ReadVariableOp*^model/decoder/dense/MatMul/ReadVariableOp,^model/encoder/conv2d/BiasAdd/ReadVariableOp+^model/encoder/conv2d/Conv2D/ReadVariableOp.^model/encoder/conv2d_1/BiasAdd/ReadVariableOp-^model/encoder/conv2d_1/Conv2D/ReadVariableOp.^model/encoder/conv2d_2/BiasAdd/ReadVariableOp-^model/encoder/conv2d_2/Conv2D/ReadVariableOp.^model/encoder/conv2d_3/BiasAdd/ReadVariableOp-^model/encoder/conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2n
5model/decoder/conv2d_transpose/BiasAdd/ReadVariableOp5model/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2ђ
>model/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp>model/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2r
7model/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp7model/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2ё
@model/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp@model/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2r
7model/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp7model/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp2ё
@model/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp@model/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2X
*model/decoder/dense/BiasAdd/ReadVariableOp*model/decoder/dense/BiasAdd/ReadVariableOp2V
)model/decoder/dense/MatMul/ReadVariableOp)model/decoder/dense/MatMul/ReadVariableOp2Z
+model/encoder/conv2d/BiasAdd/ReadVariableOp+model/encoder/conv2d/BiasAdd/ReadVariableOp2X
*model/encoder/conv2d/Conv2D/ReadVariableOp*model/encoder/conv2d/Conv2D/ReadVariableOp2^
-model/encoder/conv2d_1/BiasAdd/ReadVariableOp-model/encoder/conv2d_1/BiasAdd/ReadVariableOp2\
,model/encoder/conv2d_1/Conv2D/ReadVariableOp,model/encoder/conv2d_1/Conv2D/ReadVariableOp2^
-model/encoder/conv2d_2/BiasAdd/ReadVariableOp-model/encoder/conv2d_2/BiasAdd/ReadVariableOp2\
,model/encoder/conv2d_2/Conv2D/ReadVariableOp,model/encoder/conv2d_2/Conv2D/ReadVariableOp2^
-model/encoder/conv2d_3/BiasAdd/ReadVariableOp-model/encoder/conv2d_3/BiasAdd/ReadVariableOp2\
,model/encoder/conv2d_3/Conv2D/ReadVariableOp,model/encoder/conv2d_3/Conv2D/ReadVariableOp:X T
/
_output_shapes
:         
!
_user_specified_name	input_1
ь#
Ў
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_178325

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
Њ
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_177969

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
Љ
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_177957

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
Г
D
(__inference_reshape_layer_call_fn_179755

inputs
identityХ
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
GPU 2J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_178458h
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
┤
J
.__inference_max_pooling2d_layer_call_fn_179635

inputs
identityО
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
GPU 2J 8ѓ *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_177957Ѓ
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
ь#
Ў
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_179816

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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_179720

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
├
д
1__inference_conv2d_transpose_layer_call_fn_179778

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallч
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
GPU 2J 8ѓ *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_178325Ѕ
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
╠
_
C__inference_reshape_layer_call_and_return_conditional_losses_178458

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
И
L
0__inference_max_pooling2d_1_layer_call_fn_179665

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
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_177969Ѓ
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
╠
_
C__inference_reshape_layer_call_and_return_conditional_losses_179769

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
Ѓ
§
D__inference_conv2d_1_layer_call_and_return_conditional_losses_179660

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
ч"
љ
C__inference_encoder_layer_call_and_return_conditional_losses_178077

inputs'
conv2d_178016:
conv2d_178018:)
conv2d_1_178034: 
conv2d_1_178036: )
conv2d_2_178052: @
conv2d_2_178054:@)
conv2d_3_178070:@
conv2d_3_178072:
identityѕбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCall­
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_178016conv2d_178018*
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
GPU 2J 8ѓ *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_178015в
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_177957ў
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_178034conv2d_1_178036*
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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_178033ы
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_177969џ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_178052conv2d_2_178054*
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_178051ы
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_177981џ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_178070conv2d_3_178072*
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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_178069ч
(global_average_pooling2d/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_177994ђ
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         л
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ђ
ч
B__inference_conv2d_layer_call_and_return_conditional_losses_179630

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
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_179670

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
═
╦
$__inference_signature_wrapper_178996
input_1!
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
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_177948w
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
_user_specified_name	input_1
о 
Џ
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_179901

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
В	
О
(__inference_encoder_layer_call_fn_179315

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
C__inference_encoder_layer_call_and_return_conditional_losses_178077o
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
юЪ
Р
A__inference_model_layer_call_and_return_conditional_losses_179182

inputsG
-encoder_conv2d_conv2d_readvariableop_resource:<
.encoder_conv2d_biasadd_readvariableop_resource:I
/encoder_conv2d_1_conv2d_readvariableop_resource: >
0encoder_conv2d_1_biasadd_readvariableop_resource: I
/encoder_conv2d_2_conv2d_readvariableop_resource: @>
0encoder_conv2d_2_biasadd_readvariableop_resource:@I
/encoder_conv2d_3_conv2d_readvariableop_resource:@>
0encoder_conv2d_3_biasadd_readvariableop_resource:?
,decoder_dense_matmul_readvariableop_resource:	љ<
-decoder_dense_biasadd_readvariableop_resource:	љ[
Adecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource:F
8decoder_conv2d_transpose_biasadd_readvariableop_resource:]
Cdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_1_biasadd_readvariableop_resource:]
Cdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:H
:decoder_conv2d_transpose_2_biasadd_readvariableop_resource:
identityѕб/decoder/conv2d_transpose/BiasAdd/ReadVariableOpб8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpб1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpб:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpб1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpб:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpб$decoder/dense/BiasAdd/ReadVariableOpб#decoder/dense/MatMul/ReadVariableOpб%encoder/conv2d/BiasAdd/ReadVariableOpб$encoder/conv2d/Conv2D/ReadVariableOpб'encoder/conv2d_1/BiasAdd/ReadVariableOpб&encoder/conv2d_1/Conv2D/ReadVariableOpб'encoder/conv2d_2/BiasAdd/ReadVariableOpб&encoder/conv2d_2/Conv2D/ReadVariableOpб'encoder/conv2d_3/BiasAdd/ReadVariableOpб&encoder/conv2d_3/Conv2D/ReadVariableOpџ
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0и
encoder/conv2d/Conv2DConv2Dinputs,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
љ
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ф
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         v
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         И
encoder/max_pooling2d/MaxPoolMaxPool!encoder/conv2d/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
ъ
&encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0█
encoder/conv2d_1/Conv2DConv2D&encoder/max_pooling2d/MaxPool:output:0.encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ћ
'encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0░
encoder/conv2d_1/BiasAddBiasAdd encoder/conv2d_1/Conv2D:output:0/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          z
encoder/conv2d_1/ReluRelu!encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:          ╝
encoder/max_pooling2d_1/MaxPoolMaxPool#encoder/conv2d_1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
ъ
&encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0П
encoder/conv2d_2/Conv2DConv2D(encoder/max_pooling2d_1/MaxPool:output:0.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ћ
'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0░
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @z
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @╝
encoder/max_pooling2d_2/MaxPoolMaxPool#encoder/conv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ъ
&encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0П
encoder/conv2d_3/Conv2DConv2D(encoder/max_pooling2d_2/MaxPool:output:0.encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ћ
'encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0░
encoder/conv2d_3/BiasAddBiasAdd encoder/conv2d_3/Conv2D:output:0/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         z
encoder/conv2d_3/ReluRelu!encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:         ѕ
7encoder/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      к
%encoder/global_average_pooling2d/MeanMean#encoder/conv2d_3/Relu:activations:0@encoder/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Љ
#decoder/dense/MatMul/ReadVariableOpReadVariableOp,decoder_dense_matmul_readvariableop_resource*
_output_shapes
:	љ*
dtype0«
decoder/dense/MatMulMatMul.encoder/global_average_pooling2d/Mean:output:0+decoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љЈ
$decoder/dense/BiasAdd/ReadVariableOpReadVariableOp-decoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype0А
decoder/dense/BiasAddBiasAdddecoder/dense/MatMul:product:0,decoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љc
decoder/reshape/ShapeShapedecoder/dense/BiasAdd:output:0*
T0*
_output_shapes
:m
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :a
decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :щ
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0(decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:ц
decoder/reshape/ReshapeReshapedecoder/dense/BiasAdd:output:0&decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         n
decoder/conv2d_transpose/ShapeShape decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:v
,decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
&decoder/conv2d_transpose/strided_sliceStridedSlice'decoder/conv2d_transpose/Shape:output:05decoder/conv2d_transpose/strided_slice/stack:output:07decoder/conv2d_transpose/strided_slice/stack_1:output:07decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :є
decoder/conv2d_transpose/stackPack/decoder/conv2d_transpose/strided_slice:output:0)decoder/conv2d_transpose/stack/1:output:0)decoder/conv2d_transpose/stack/2:output:0)decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:x
.decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
(decoder/conv2d_transpose/strided_slice_1StridedSlice'decoder/conv2d_transpose/stack:output:07decoder/conv2d_transpose/strided_slice_1/stack:output:09decoder/conv2d_transpose/strided_slice_1/stack_1:output:09decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┬
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0░
)decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput'decoder/conv2d_transpose/stack:output:0@decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0 decoder/reshape/Reshape:output:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
ц
/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0м
 decoder/conv2d_transpose/BiasAddBiasAdd2decoder/conv2d_transpose/conv2d_transpose:output:07decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         і
decoder/conv2d_transpose/ReluRelu)decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:         {
 decoder/conv2d_transpose_1/ShapeShape+decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
(decoder/conv2d_transpose_1/strided_sliceStridedSlice)decoder/conv2d_transpose_1/Shape:output:07decoder/conv2d_transpose_1/strided_slice/stack:output:09decoder/conv2d_transpose_1/strided_slice/stack_1:output:09decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :љ
 decoder/conv2d_transpose_1/stackPack1decoder/conv2d_transpose_1/strided_slice:output:0+decoder/conv2d_transpose_1/stack/1:output:0+decoder/conv2d_transpose_1/stack/2:output:0+decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
*decoder/conv2d_transpose_1/strided_slice_1StridedSlice)decoder/conv2d_transpose_1/stack:output:09decoder/conv2d_transpose_1/strided_slice_1/stack:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskк
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0└
+decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_1/stack:output:0Bdecoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0+decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
е
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
"decoder/conv2d_transpose_1/BiasAddBiasAdd4decoder/conv2d_transpose_1/conv2d_transpose:output:09decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ј
decoder/conv2d_transpose_1/ReluRelu+decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:         }
 decoder/conv2d_transpose_2/ShapeShape-decoder/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
(decoder/conv2d_transpose_2/strided_sliceStridedSlice)decoder/conv2d_transpose_2/Shape:output:07decoder/conv2d_transpose_2/strided_slice/stack:output:09decoder/conv2d_transpose_2/strided_slice/stack_1:output:09decoder/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :љ
 decoder/conv2d_transpose_2/stackPack1decoder/conv2d_transpose_2/strided_slice:output:0+decoder/conv2d_transpose_2/stack/1:output:0+decoder/conv2d_transpose_2/stack/2:output:0+decoder/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
*decoder/conv2d_transpose_2/strided_slice_1StridedSlice)decoder/conv2d_transpose_2/stack:output:09decoder/conv2d_transpose_2/strided_slice_1/stack:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskк
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0┬
+decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_2/stack:output:0Bdecoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
е
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0п
"decoder/conv2d_transpose_2/BiasAddBiasAdd4decoder/conv2d_transpose_2/conv2d_transpose:output:09decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ѓ
IdentityIdentity+decoder/conv2d_transpose_2/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         ф
NoOpNoOp0^decoder/conv2d_transpose/BiasAdd/ReadVariableOp9^decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp%^decoder/dense/BiasAdd/ReadVariableOp$^decoder/dense/MatMul/ReadVariableOp&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp(^encoder/conv2d_3/BiasAdd/ReadVariableOp'^encoder/conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         : : : : : : : : : : : : : : : : 2b
/decoder/conv2d_transpose/BiasAdd/ReadVariableOp/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2t
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2L
$decoder/dense/BiasAdd/ReadVariableOp$decoder/dense/BiasAdd/ReadVariableOp2J
#decoder/dense/MatMul/ReadVariableOp#decoder/dense/MatMul/ReadVariableOp2N
%encoder/conv2d/BiasAdd/ReadVariableOp%encoder/conv2d/BiasAdd/ReadVariableOp2L
$encoder/conv2d/Conv2D/ReadVariableOp$encoder/conv2d/Conv2D/ReadVariableOp2R
'encoder/conv2d_1/BiasAdd/ReadVariableOp'encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&encoder/conv2d_1/Conv2D/ReadVariableOp&encoder/conv2d_1/Conv2D/ReadVariableOp2R
'encoder/conv2d_2/BiasAdd/ReadVariableOp'encoder/conv2d_2/BiasAdd/ReadVariableOp2P
&encoder/conv2d_2/Conv2D/ReadVariableOp&encoder/conv2d_2/Conv2D/ReadVariableOp2R
'encoder/conv2d_3/BiasAdd/ReadVariableOp'encoder/conv2d_3/BiasAdd/ReadVariableOp2P
&encoder/conv2d_3/Conv2D/ReadVariableOp&encoder/conv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Љ
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_179640

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
яp
Ч
__inference__traced_save_180098
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_nadam_conv2d_kernel_m_read_readvariableop2
.savev2_nadam_conv2d_bias_m_read_readvariableop6
2savev2_nadam_conv2d_1_kernel_m_read_readvariableop4
0savev2_nadam_conv2d_1_bias_m_read_readvariableop6
2savev2_nadam_conv2d_2_kernel_m_read_readvariableop4
0savev2_nadam_conv2d_2_bias_m_read_readvariableop6
2savev2_nadam_conv2d_3_kernel_m_read_readvariableop4
0savev2_nadam_conv2d_3_bias_m_read_readvariableop3
/savev2_nadam_dense_kernel_m_read_readvariableop1
-savev2_nadam_dense_bias_m_read_readvariableop>
:savev2_nadam_conv2d_transpose_kernel_m_read_readvariableop<
8savev2_nadam_conv2d_transpose_bias_m_read_readvariableop@
<savev2_nadam_conv2d_transpose_1_kernel_m_read_readvariableop>
:savev2_nadam_conv2d_transpose_1_bias_m_read_readvariableop@
<savev2_nadam_conv2d_transpose_2_kernel_m_read_readvariableop>
:savev2_nadam_conv2d_transpose_2_bias_m_read_readvariableop4
0savev2_nadam_conv2d_kernel_v_read_readvariableop2
.savev2_nadam_conv2d_bias_v_read_readvariableop6
2savev2_nadam_conv2d_1_kernel_v_read_readvariableop4
0savev2_nadam_conv2d_1_bias_v_read_readvariableop6
2savev2_nadam_conv2d_2_kernel_v_read_readvariableop4
0savev2_nadam_conv2d_2_bias_v_read_readvariableop6
2savev2_nadam_conv2d_3_kernel_v_read_readvariableop4
0savev2_nadam_conv2d_3_bias_v_read_readvariableop3
/savev2_nadam_dense_kernel_v_read_readvariableop1
-savev2_nadam_dense_bias_v_read_readvariableop>
:savev2_nadam_conv2d_transpose_kernel_v_read_readvariableop<
8savev2_nadam_conv2d_transpose_bias_v_read_readvariableop@
<savev2_nadam_conv2d_transpose_1_kernel_v_read_readvariableop>
:savev2_nadam_conv2d_transpose_1_bias_v_read_readvariableop@
<savev2_nadam_conv2d_transpose_2_kernel_v_read_readvariableop>
:savev2_nadam_conv2d_transpose_2_bias_v_read_readvariableop
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
valueђB~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B њ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_nadam_conv2d_kernel_m_read_readvariableop.savev2_nadam_conv2d_bias_m_read_readvariableop2savev2_nadam_conv2d_1_kernel_m_read_readvariableop0savev2_nadam_conv2d_1_bias_m_read_readvariableop2savev2_nadam_conv2d_2_kernel_m_read_readvariableop0savev2_nadam_conv2d_2_bias_m_read_readvariableop2savev2_nadam_conv2d_3_kernel_m_read_readvariableop0savev2_nadam_conv2d_3_bias_m_read_readvariableop/savev2_nadam_dense_kernel_m_read_readvariableop-savev2_nadam_dense_bias_m_read_readvariableop:savev2_nadam_conv2d_transpose_kernel_m_read_readvariableop8savev2_nadam_conv2d_transpose_bias_m_read_readvariableop<savev2_nadam_conv2d_transpose_1_kernel_m_read_readvariableop:savev2_nadam_conv2d_transpose_1_bias_m_read_readvariableop<savev2_nadam_conv2d_transpose_2_kernel_m_read_readvariableop:savev2_nadam_conv2d_transpose_2_bias_m_read_readvariableop0savev2_nadam_conv2d_kernel_v_read_readvariableop.savev2_nadam_conv2d_bias_v_read_readvariableop2savev2_nadam_conv2d_1_kernel_v_read_readvariableop0savev2_nadam_conv2d_1_bias_v_read_readvariableop2savev2_nadam_conv2d_2_kernel_v_read_readvariableop0savev2_nadam_conv2d_2_bias_v_read_readvariableop2savev2_nadam_conv2d_3_kernel_v_read_readvariableop0savev2_nadam_conv2d_3_bias_v_read_readvariableop/savev2_nadam_dense_kernel_v_read_readvariableop-savev2_nadam_dense_bias_v_read_readvariableop:savev2_nadam_conv2d_transpose_kernel_v_read_readvariableop8savev2_nadam_conv2d_transpose_bias_v_read_readvariableop<savev2_nadam_conv2d_transpose_1_kernel_v_read_readvariableop:savev2_nadam_conv2d_transpose_1_bias_v_read_readvariableop<savev2_nadam_conv2d_transpose_2_kernel_v_read_readvariableop:savev2_nadam_conv2d_transpose_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
╔
я
A__inference_model_layer_call_and_return_conditional_losses_178691

inputs(
encoder_178656:
encoder_178658:(
encoder_178660: 
encoder_178662: (
encoder_178664: @
encoder_178666:@(
encoder_178668:@
encoder_178670:!
decoder_178673:	љ
decoder_178675:	љ(
decoder_178677:
decoder_178679:(
decoder_178681:
decoder_178683:(
decoder_178685:
decoder_178687:
identityѕбdecoder/StatefulPartitionedCallбencoder/StatefulPartitionedCallп
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_178656encoder_178658encoder_178660encoder_178662encoder_178664encoder_178666encoder_178668encoder_178670*
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
C__inference_encoder_layer_call_and_return_conditional_losses_178077ѓ
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_178673decoder_178675decoder_178677decoder_178679decoder_178681decoder_178683decoder_178685decoder_178687*
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
C__inference_decoder_layer_call_and_return_conditional_losses_178476
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
Х-
м
C__inference_encoder_layer_call_and_return_conditional_losses_179373

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@6
(conv2d_3_biasadd_readvariableop_resource:
identityѕбconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpбconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpбconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpі
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Д
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ђ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0њ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         е
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
ј
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0├
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
ё
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:          г
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
ј
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0┼
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ё
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ў
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @г
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ј
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0┼
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ё
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ў
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:         ђ
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      «
global_average_pooling2d/MeanMeanconv2d_3/Relu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         u
IdentityIdentity&global_average_pooling2d/Mean:output:0^NoOp*
T0*'
_output_shapes
:         ╬
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
№
═
&__inference_model_layer_call_fn_178875
input_1!
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
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8ѓ *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_178803w
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
_user_specified_name	input_1
Ѓ
§
D__inference_conv2d_2_layer_call_and_return_conditional_losses_179690

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
ч"
љ
C__inference_encoder_layer_call_and_return_conditional_losses_178187

inputs'
conv2d_178162:
conv2d_178164:)
conv2d_1_178168: 
conv2d_1_178170: )
conv2d_2_178174: @
conv2d_2_178176:@)
conv2d_3_178180:@
conv2d_3_178182:
identityѕбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCall­
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_178162conv2d_178164*
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
GPU 2J 8ѓ *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_178015в
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_177957ў
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_178168conv2d_1_178170*
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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_178033ы
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_177969џ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_178174conv2d_2_178176*
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_178051ы
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_177981џ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_178180conv2d_3_178182*
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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_178069ч
(global_average_pooling2d/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_177994ђ
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         л
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:         : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
К
е
3__inference_conv2d_transpose_1_layer_call_fn_179825

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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_178370Ѕ
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
№	
п
(__inference_encoder_layer_call_fn_178096
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@
	unknown_6:
identityѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
C__inference_encoder_layer_call_and_return_conditional_losses_178077o
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
_user_specified_name	input_1
╠
▀
A__inference_model_layer_call_and_return_conditional_losses_178951
input_1(
encoder_178916:
encoder_178918:(
encoder_178920: 
encoder_178922: (
encoder_178924: @
encoder_178926:@(
encoder_178928:@
encoder_178930:!
decoder_178933:	љ
decoder_178935:	љ(
decoder_178937:
decoder_178939:(
decoder_178941:
decoder_178943:(
decoder_178945:
decoder_178947:
identityѕбdecoder/StatefulPartitionedCallбencoder/StatefulPartitionedCall┘
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_178916encoder_178918encoder_178920encoder_178922encoder_178924encoder_178926encoder_178928encoder_178930*
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
C__inference_encoder_layer_call_and_return_conditional_losses_178187ѓ
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_178933decoder_178935decoder_178937decoder_178939decoder_178941decoder_178943decoder_178945decoder_178947*
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
C__inference_decoder_layer_call_and_return_conditional_losses_178559
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
_user_specified_name	input_1
В
╠
&__inference_model_layer_call_fn_179033

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
identityѕбStatefulPartitionedCallЎ
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
GPU 2J 8ѓ *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_178691w
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
»С
Л%
"__inference__traced_restore_180282
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel: .
 assignvariableop_3_conv2d_1_bias: <
"assignvariableop_4_conv2d_2_kernel: @.
 assignvariableop_5_conv2d_2_bias:@<
"assignvariableop_6_conv2d_3_kernel:@.
 assignvariableop_7_conv2d_3_bias:2
assignvariableop_8_dense_kernel:	љ,
assignvariableop_9_dense_bias:	љE
+assignvariableop_10_conv2d_transpose_kernel:7
)assignvariableop_11_conv2d_transpose_bias:G
-assignvariableop_12_conv2d_transpose_1_kernel:9
+assignvariableop_13_conv2d_transpose_1_bias:G
-assignvariableop_14_conv2d_transpose_2_kernel:9
+assignvariableop_15_conv2d_transpose_2_bias:(
assignvariableop_16_nadam_iter:	 *
 assignvariableop_17_nadam_beta_1: *
 assignvariableop_18_nadam_beta_2: )
assignvariableop_19_nadam_decay: 1
'assignvariableop_20_nadam_learning_rate: 2
(assignvariableop_21_nadam_momentum_cache: %
assignvariableop_22_total_1: %
assignvariableop_23_count_1: #
assignvariableop_24_total: #
assignvariableop_25_count: C
)assignvariableop_26_nadam_conv2d_kernel_m:5
'assignvariableop_27_nadam_conv2d_bias_m:E
+assignvariableop_28_nadam_conv2d_1_kernel_m: 7
)assignvariableop_29_nadam_conv2d_1_bias_m: E
+assignvariableop_30_nadam_conv2d_2_kernel_m: @7
)assignvariableop_31_nadam_conv2d_2_bias_m:@E
+assignvariableop_32_nadam_conv2d_3_kernel_m:@7
)assignvariableop_33_nadam_conv2d_3_bias_m:;
(assignvariableop_34_nadam_dense_kernel_m:	љ5
&assignvariableop_35_nadam_dense_bias_m:	љM
3assignvariableop_36_nadam_conv2d_transpose_kernel_m:?
1assignvariableop_37_nadam_conv2d_transpose_bias_m:O
5assignvariableop_38_nadam_conv2d_transpose_1_kernel_m:A
3assignvariableop_39_nadam_conv2d_transpose_1_bias_m:O
5assignvariableop_40_nadam_conv2d_transpose_2_kernel_m:A
3assignvariableop_41_nadam_conv2d_transpose_2_bias_m:C
)assignvariableop_42_nadam_conv2d_kernel_v:5
'assignvariableop_43_nadam_conv2d_bias_v:E
+assignvariableop_44_nadam_conv2d_1_kernel_v: 7
)assignvariableop_45_nadam_conv2d_1_bias_v: E
+assignvariableop_46_nadam_conv2d_2_kernel_v: @7
)assignvariableop_47_nadam_conv2d_2_bias_v:@E
+assignvariableop_48_nadam_conv2d_3_kernel_v:@7
)assignvariableop_49_nadam_conv2d_3_bias_v:;
(assignvariableop_50_nadam_dense_kernel_v:	љ5
&assignvariableop_51_nadam_dense_bias_v:	љM
3assignvariableop_52_nadam_conv2d_transpose_kernel_v:?
1assignvariableop_53_nadam_conv2d_transpose_bias_v:O
5assignvariableop_54_nadam_conv2d_transpose_1_kernel_v:A
3assignvariableop_55_nadam_conv2d_transpose_1_bias_v:O
5assignvariableop_56_nadam_conv2d_transpose_2_kernel_v:A
3assignvariableop_57_nadam_conv2d_transpose_2_bias_v:
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
:Ѕ
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_10AssignVariableOp+assignvariableop_10_conv2d_transpose_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_11AssignVariableOp)assignvariableop_11_conv2d_transpose_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_12AssignVariableOp-assignvariableop_12_conv2d_transpose_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_13AssignVariableOp+assignvariableop_13_conv2d_transpose_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_14AssignVariableOp-assignvariableop_14_conv2d_transpose_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_15AssignVariableOp+assignvariableop_15_conv2d_transpose_2_biasIdentity_15:output:0"/device:CPU:0*
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
:џ
AssignVariableOp_26AssignVariableOp)assignvariableop_26_nadam_conv2d_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_27AssignVariableOp'assignvariableop_27_nadam_conv2d_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_28AssignVariableOp+assignvariableop_28_nadam_conv2d_1_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_29AssignVariableOp)assignvariableop_29_nadam_conv2d_1_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_30AssignVariableOp+assignvariableop_30_nadam_conv2d_2_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_31AssignVariableOp)assignvariableop_31_nadam_conv2d_2_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_32AssignVariableOp+assignvariableop_32_nadam_conv2d_3_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_33AssignVariableOp)assignvariableop_33_nadam_conv2d_3_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_34AssignVariableOp(assignvariableop_34_nadam_dense_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_35AssignVariableOp&assignvariableop_35_nadam_dense_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_36AssignVariableOp3assignvariableop_36_nadam_conv2d_transpose_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_37AssignVariableOp1assignvariableop_37_nadam_conv2d_transpose_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_38AssignVariableOp5assignvariableop_38_nadam_conv2d_transpose_1_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_39AssignVariableOp3assignvariableop_39_nadam_conv2d_transpose_1_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_40AssignVariableOp5assignvariableop_40_nadam_conv2d_transpose_2_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_41AssignVariableOp3assignvariableop_41_nadam_conv2d_transpose_2_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_42AssignVariableOp)assignvariableop_42_nadam_conv2d_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_43AssignVariableOp'assignvariableop_43_nadam_conv2d_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_44AssignVariableOp+assignvariableop_44_nadam_conv2d_1_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_45AssignVariableOp)assignvariableop_45_nadam_conv2d_1_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_46AssignVariableOp+assignvariableop_46_nadam_conv2d_2_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_47AssignVariableOp)assignvariableop_47_nadam_conv2d_2_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_48AssignVariableOp+assignvariableop_48_nadam_conv2d_3_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_49AssignVariableOp)assignvariableop_49_nadam_conv2d_3_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_50AssignVariableOp(assignvariableop_50_nadam_dense_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_51AssignVariableOp&assignvariableop_51_nadam_dense_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_52AssignVariableOp3assignvariableop_52_nadam_conv2d_transpose_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_53AssignVariableOp1assignvariableop_53_nadam_conv2d_transpose_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_54AssignVariableOp5assignvariableop_54_nadam_conv2d_transpose_1_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_55AssignVariableOp3assignvariableop_55_nadam_conv2d_transpose_1_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_56AssignVariableOp5assignvariableop_56_nadam_conv2d_transpose_2_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:ц
AssignVariableOp_57AssignVariableOp3assignvariableop_57_nadam_conv2d_transpose_2_bias_vIdentity_57:output:0"/device:CPU:0*
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
Ѓ
§
D__inference_conv2d_2_layer_call_and_return_conditional_losses_178051

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
input_18
serving_default_input_1:0         C
decoder8
StatefulPartitionedCall:0         tensorflow/serving/predict:█ї
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
═
<trace_0
=trace_1
>trace_2
?trace_32Р
&__inference_model_layer_call_fn_178726
&__inference_model_layer_call_fn_179033
&__inference_model_layer_call_fn_179070
&__inference_model_layer_call_fn_178875┐
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
╣
@trace_0
Atrace_1
Btrace_2
Ctrace_32╬
A__inference_model_layer_call_and_return_conditional_losses_179182
A__inference_model_layer_call_and_return_conditional_losses_179294
A__inference_model_layer_call_and_return_conditional_losses_178913
A__inference_model_layer_call_and_return_conditional_losses_178951┐
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
!__inference__wrapped_model_177948input_1"ў
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
(__inference_encoder_layer_call_fn_178096
(__inference_encoder_layer_call_fn_179315
(__inference_encoder_layer_call_fn_179336
(__inference_encoder_layer_call_fn_178227┐
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
C__inference_encoder_layer_call_and_return_conditional_losses_179373
C__inference_encoder_layer_call_and_return_conditional_losses_179410
C__inference_encoder_layer_call_and_return_conditional_losses_178255
C__inference_encoder_layer_call_and_return_conditional_losses_178283┐
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
(__inference_decoder_layer_call_fn_178495
(__inference_decoder_layer_call_fn_179431
(__inference_decoder_layer_call_fn_179452
(__inference_decoder_layer_call_fn_178599┐
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
C__inference_decoder_layer_call_and_return_conditional_losses_179531
C__inference_decoder_layer_call_and_return_conditional_losses_179610
C__inference_decoder_layer_call_and_return_conditional_losses_178624
C__inference_decoder_layer_call_and_return_conditional_losses_178649┐
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
':%2conv2d/kernel
:2conv2d/bias
):' 2conv2d_1/kernel
: 2conv2d_1/bias
):' @2conv2d_2/kernel
:@2conv2d_2/bias
):'@2conv2d_3/kernel
:2conv2d_3/bias
:	љ2dense/kernel
:љ2
dense/bias
1:/2conv2d_transpose/kernel
#:!2conv2d_transpose/bias
3:12conv2d_transpose_1/kernel
%:#2conv2d_transpose_1/bias
3:12conv2d_transpose_2/kernel
%:#2conv2d_transpose_2/bias
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
ЭBш
&__inference_model_layer_call_fn_178726input_1"┐
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
эBЗ
&__inference_model_layer_call_fn_179033inputs"┐
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
эBЗ
&__inference_model_layer_call_fn_179070inputs"┐
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
ЭBш
&__inference_model_layer_call_fn_178875input_1"┐
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
њBЈ
A__inference_model_layer_call_and_return_conditional_losses_179182inputs"┐
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
њBЈ
A__inference_model_layer_call_and_return_conditional_losses_179294inputs"┐
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
ЊBљ
A__inference_model_layer_call_and_return_conditional_losses_178913input_1"┐
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
ЊBљ
A__inference_model_layer_call_and_return_conditional_losses_178951input_1"┐
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
$__inference_signature_wrapper_178996input_1"ћ
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
ь
┴trace_02╬
'__inference_conv2d_layer_call_fn_179619б
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
ѕ
┬trace_02ж
B__inference_conv2d_layer_call_and_return_conditional_losses_179630б
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
З
╚trace_02Н
.__inference_max_pooling2d_layer_call_fn_179635б
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
Ј
╔trace_02­
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_179640б
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
)__inference_conv2d_1_layer_call_fn_179649б
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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_179660б
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
0__inference_max_pooling2d_1_layer_call_fn_179665б
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
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_179670б
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
)__inference_conv2d_2_layer_call_fn_179679б
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_179690б
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
0__inference_max_pooling2d_2_layer_call_fn_179695б
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
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_179700б
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
)__inference_conv2d_3_layer_call_fn_179709б
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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_179720б
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
 
Ыtrace_02Я
9__inference_global_average_pooling2d_layer_call_fn_179725б
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
џ
зtrace_02ч
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_179731б
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
(__inference_encoder_layer_call_fn_178096input_1"┐
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
(__inference_encoder_layer_call_fn_179315inputs"┐
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
(__inference_encoder_layer_call_fn_179336inputs"┐
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
(__inference_encoder_layer_call_fn_178227input_1"┐
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
C__inference_encoder_layer_call_and_return_conditional_losses_179373inputs"┐
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
C__inference_encoder_layer_call_and_return_conditional_losses_179410inputs"┐
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
C__inference_encoder_layer_call_and_return_conditional_losses_178255input_1"┐
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
C__inference_encoder_layer_call_and_return_conditional_losses_178283input_1"┐
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
В
щtrace_02═
&__inference_dense_layer_call_fn_179740б
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
Є
Щtrace_02У
A__inference_dense_layer_call_and_return_conditional_losses_179750б
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
Ь
ђtrace_02¤
(__inference_reshape_layer_call_fn_179755б
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
Ѕ
Ђtrace_02Ж
C__inference_reshape_layer_call_and_return_conditional_losses_179769б
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
э
Єtrace_02п
1__inference_conv2d_transpose_layer_call_fn_179778б
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
њ
ѕtrace_02з
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_179816б
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
3__inference_conv2d_transpose_1_layer_call_fn_179825б
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_179859б
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
3__inference_conv2d_transpose_2_layer_call_fn_179868б
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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_179901б
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
(__inference_decoder_layer_call_fn_178495input_2"┐
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
(__inference_decoder_layer_call_fn_179431inputs"┐
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
(__inference_decoder_layer_call_fn_179452inputs"┐
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
(__inference_decoder_layer_call_fn_178599input_2"┐
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
C__inference_decoder_layer_call_and_return_conditional_losses_179531inputs"┐
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
C__inference_decoder_layer_call_and_return_conditional_losses_179610inputs"┐
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
C__inference_decoder_layer_call_and_return_conditional_losses_178624input_2"┐
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
C__inference_decoder_layer_call_and_return_conditional_losses_178649input_2"┐
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
█Bп
'__inference_conv2d_layer_call_fn_179619inputs"б
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
ШBз
B__inference_conv2d_layer_call_and_return_conditional_losses_179630inputs"б
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
РB▀
.__inference_max_pooling2d_layer_call_fn_179635inputs"б
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
§BЩ
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_179640inputs"б
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
)__inference_conv2d_1_layer_call_fn_179649inputs"б
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
D__inference_conv2d_1_layer_call_and_return_conditional_losses_179660inputs"б
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
0__inference_max_pooling2d_1_layer_call_fn_179665inputs"б
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
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_179670inputs"б
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
)__inference_conv2d_2_layer_call_fn_179679inputs"б
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
D__inference_conv2d_2_layer_call_and_return_conditional_losses_179690inputs"б
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
0__inference_max_pooling2d_2_layer_call_fn_179695inputs"б
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
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_179700inputs"б
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
)__inference_conv2d_3_layer_call_fn_179709inputs"б
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
D__inference_conv2d_3_layer_call_and_return_conditional_losses_179720inputs"б
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
ьBЖ
9__inference_global_average_pooling2d_layer_call_fn_179725inputs"б
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
ѕBЁ
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_179731inputs"б
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
┌BО
&__inference_dense_layer_call_fn_179740inputs"б
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
шBЫ
A__inference_dense_layer_call_and_return_conditional_losses_179750inputs"б
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
(__inference_reshape_layer_call_fn_179755inputs"б
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
C__inference_reshape_layer_call_and_return_conditional_losses_179769inputs"б
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
тBР
1__inference_conv2d_transpose_layer_call_fn_179778inputs"б
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
ђB§
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_179816inputs"б
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
3__inference_conv2d_transpose_1_layer_call_fn_179825inputs"б
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_179859inputs"б
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
3__inference_conv2d_transpose_2_layer_call_fn_179868inputs"б
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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_179901inputs"б
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
-:+2Nadam/conv2d/kernel/m
:2Nadam/conv2d/bias/m
/:- 2Nadam/conv2d_1/kernel/m
!: 2Nadam/conv2d_1/bias/m
/:- @2Nadam/conv2d_2/kernel/m
!:@2Nadam/conv2d_2/bias/m
/:-@2Nadam/conv2d_3/kernel/m
!:2Nadam/conv2d_3/bias/m
%:#	љ2Nadam/dense/kernel/m
:љ2Nadam/dense/bias/m
7:52Nadam/conv2d_transpose/kernel/m
):'2Nadam/conv2d_transpose/bias/m
9:72!Nadam/conv2d_transpose_1/kernel/m
+:)2Nadam/conv2d_transpose_1/bias/m
9:72!Nadam/conv2d_transpose_2/kernel/m
+:)2Nadam/conv2d_transpose_2/bias/m
-:+2Nadam/conv2d/kernel/v
:2Nadam/conv2d/bias/v
/:- 2Nadam/conv2d_1/kernel/v
!: 2Nadam/conv2d_1/bias/v
/:- @2Nadam/conv2d_2/kernel/v
!:@2Nadam/conv2d_2/bias/v
/:-@2Nadam/conv2d_3/kernel/v
!:2Nadam/conv2d_3/bias/v
%:#	љ2Nadam/dense/kernel/v
:љ2Nadam/dense/bias/v
7:52Nadam/conv2d_transpose/kernel/v
):'2Nadam/conv2d_transpose/bias/v
9:72!Nadam/conv2d_transpose_1/kernel/v
+:)2Nadam/conv2d_transpose_1/bias/v
9:72!Nadam/conv2d_transpose_2/kernel/v
+:)2Nadam/conv2d_transpose_2/bias/vГ
!__inference__wrapped_model_177948Є'()*+,-./01234568б5
.б+
)і&
input_1         
ф "9ф6
4
decoder)і&
decoder         ┤
D__inference_conv2d_1_layer_call_and_return_conditional_losses_179660l)*7б4
-б*
(і%
inputs         
ф "-б*
#і 
0          
џ ї
)__inference_conv2d_1_layer_call_fn_179649_)*7б4
-б*
(і%
inputs         
ф " і          ┤
D__inference_conv2d_2_layer_call_and_return_conditional_losses_179690l+,7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         @
џ ї
)__inference_conv2d_2_layer_call_fn_179679_+,7б4
-б*
(і%
inputs          
ф " і         @┤
D__inference_conv2d_3_layer_call_and_return_conditional_losses_179720l-.7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         
џ ї
)__inference_conv2d_3_layer_call_fn_179709_-.7б4
-б*
(і%
inputs         @
ф " і         ▓
B__inference_conv2d_layer_call_and_return_conditional_losses_179630l'(7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ і
'__inference_conv2d_layer_call_fn_179619_'(7б4
-б*
(і%
inputs         
ф " і         с
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_179859љ34IбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ ╗
3__inference_conv2d_transpose_1_layer_call_fn_179825Ѓ34IбF
?б<
:і7
inputs+                           
ф "2і/+                           с
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_179901љ56IбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ ╗
3__inference_conv2d_transpose_2_layer_call_fn_179868Ѓ56IбF
?б<
:і7
inputs+                           
ф "2і/+                           р
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_179816љ12IбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ ╣
1__inference_conv2d_transpose_layer_call_fn_179778Ѓ12IбF
?б<
:і7
inputs+                           
ф "2і/+                           ║
C__inference_decoder_layer_call_and_return_conditional_losses_178624s/01234568б5
.б+
!і
input_2         
p 

 
ф "-б*
#і 
0         
џ ║
C__inference_decoder_layer_call_and_return_conditional_losses_178649s/01234568б5
.б+
!і
input_2         
p

 
ф "-б*
#і 
0         
џ ╣
C__inference_decoder_layer_call_and_return_conditional_losses_179531r/01234567б4
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
C__inference_decoder_layer_call_and_return_conditional_losses_179610r/01234567б4
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
(__inference_decoder_layer_call_fn_178495f/01234568б5
.б+
!і
input_2         
p 

 
ф " і         њ
(__inference_decoder_layer_call_fn_178599f/01234568б5
.б+
!і
input_2         
p

 
ф " і         Љ
(__inference_decoder_layer_call_fn_179431e/01234567б4
-б*
 і
inputs         
p 

 
ф " і         Љ
(__inference_decoder_layer_call_fn_179452e/01234567б4
-б*
 і
inputs         
p

 
ф " і         б
A__inference_dense_layer_call_and_return_conditional_losses_179750]/0/б,
%б"
 і
inputs         
ф "&б#
і
0         љ
џ z
&__inference_dense_layer_call_fn_179740P/0/б,
%б"
 і
inputs         
ф "і         љ║
C__inference_encoder_layer_call_and_return_conditional_losses_178255s'()*+,-.@б=
6б3
)і&
input_1         
p 

 
ф "%б"
і
0         
џ ║
C__inference_encoder_layer_call_and_return_conditional_losses_178283s'()*+,-.@б=
6б3
)і&
input_1         
p

 
ф "%б"
і
0         
џ ╣
C__inference_encoder_layer_call_and_return_conditional_losses_179373r'()*+,-.?б<
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
C__inference_encoder_layer_call_and_return_conditional_losses_179410r'()*+,-.?б<
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
(__inference_encoder_layer_call_fn_178096f'()*+,-.@б=
6б3
)і&
input_1         
p 

 
ф "і         њ
(__inference_encoder_layer_call_fn_178227f'()*+,-.@б=
6б3
)і&
input_1         
p

 
ф "і         Љ
(__inference_encoder_layer_call_fn_179315e'()*+,-.?б<
5б2
(і%
inputs         
p 

 
ф "і         Љ
(__inference_encoder_layer_call_fn_179336e'()*+,-.?б<
5б2
(і%
inputs         
p

 
ф "і         П
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_179731ёRбO
HбE
Cі@
inputs4                                    
ф ".б+
$і!
0                  
џ ┤
9__inference_global_average_pooling2d_layer_call_fn_179725wRбO
HбE
Cі@
inputs4                                    
ф "!і                  Ь
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_179670ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_1_layer_call_fn_179665ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ь
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_179700ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_2_layer_call_fn_179695ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    В
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_179640ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ─
.__inference_max_pooling2d_layer_call_fn_179635ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ╔
A__inference_model_layer_call_and_return_conditional_losses_178913Ѓ'()*+,-./0123456@б=
6б3
)і&
input_1         
p 

 
ф "-б*
#і 
0         
џ ╔
A__inference_model_layer_call_and_return_conditional_losses_178951Ѓ'()*+,-./0123456@б=
6б3
)і&
input_1         
p

 
ф "-б*
#і 
0         
џ ╚
A__inference_model_layer_call_and_return_conditional_losses_179182ѓ'()*+,-./0123456?б<
5б2
(і%
inputs         
p 

 
ф "-б*
#і 
0         
џ ╚
A__inference_model_layer_call_and_return_conditional_losses_179294ѓ'()*+,-./0123456?б<
5б2
(і%
inputs         
p

 
ф "-б*
#і 
0         
џ а
&__inference_model_layer_call_fn_178726v'()*+,-./0123456@б=
6б3
)і&
input_1         
p 

 
ф " і         а
&__inference_model_layer_call_fn_178875v'()*+,-./0123456@б=
6б3
)і&
input_1         
p

 
ф " і         Ъ
&__inference_model_layer_call_fn_179033u'()*+,-./0123456?б<
5б2
(і%
inputs         
p 

 
ф " і         Ъ
&__inference_model_layer_call_fn_179070u'()*+,-./0123456?б<
5б2
(і%
inputs         
p

 
ф " і         е
C__inference_reshape_layer_call_and_return_conditional_losses_179769a0б-
&б#
!і
inputs         љ
ф "-б*
#і 
0         
џ ђ
(__inference_reshape_layer_call_fn_179755T0б-
&б#
!і
inputs         љ
ф " і         ╗
$__inference_signature_wrapper_178996њ'()*+,-./0123456Cб@
б 
9ф6
4
input_1)і&
input_1         "9ф6
4
decoder)і&
decoder         