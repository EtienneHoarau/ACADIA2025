Ф
ЭА
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

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
.
Identity

input"T
output"T"	
Ttype
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
 "serve*2.10.02unknown8ѕ№

Adam/conv2d_transpose_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_13/bias/v

3Adam/conv2d_transpose_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_13/bias/v*
_output_shapes
:*
dtype0
І
!Adam/conv2d_transpose_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_13/kernel/v

5Adam/conv2d_transpose_13/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_13/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_transpose_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_12/bias/v

3Adam/conv2d_transpose_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_12/bias/v*
_output_shapes
: *
dtype0
І
!Adam/conv2d_transpose_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_12/kernel/v

5Adam/conv2d_transpose_12/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_12/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_transpose_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/conv2d_transpose_11/bias/v

3Adam/conv2d_transpose_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_11/bias/v*
_output_shapes
:@*
dtype0
Ї
!Adam/conv2d_transpose_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/conv2d_transpose_11/kernel/v
 
5Adam/conv2d_transpose_11/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_11/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_transpose_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_10/bias/v

3Adam/conv2d_transpose_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_10/bias/v*
_output_shapes	
:*
dtype0
Ј
!Adam/conv2d_transpose_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_10/kernel/v
Ё
5Adam/conv2d_transpose_10/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_10/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_19/bias/v
|
)Adam/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_19/kernel/v

+Adam/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_18/bias/v
|
)Adam/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_18/kernel/v

+Adam/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_17/bias/v
{
)Adam/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_17/kernel/v

+Adam/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_16/bias/v
{
)Adam/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_16/kernel/v

+Adam/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_transpose_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_13/bias/m

3Adam/conv2d_transpose_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_13/bias/m*
_output_shapes
:*
dtype0
І
!Adam/conv2d_transpose_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_13/kernel/m

5Adam/conv2d_transpose_13/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_13/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_transpose_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_12/bias/m

3Adam/conv2d_transpose_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_12/bias/m*
_output_shapes
: *
dtype0
І
!Adam/conv2d_transpose_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_12/kernel/m

5Adam/conv2d_transpose_12/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_12/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_transpose_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/conv2d_transpose_11/bias/m

3Adam/conv2d_transpose_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_11/bias/m*
_output_shapes
:@*
dtype0
Ї
!Adam/conv2d_transpose_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/conv2d_transpose_11/kernel/m
 
5Adam/conv2d_transpose_11/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_11/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_transpose_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_10/bias/m

3Adam/conv2d_transpose_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_10/bias/m*
_output_shapes	
:*
dtype0
Ј
!Adam/conv2d_transpose_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_10/kernel/m
Ё
5Adam/conv2d_transpose_10/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_10/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_19/bias/m
|
)Adam/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_19/kernel/m

+Adam/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_18/bias/m
|
)Adam/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_18/kernel/m

+Adam/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_17/bias/m
{
)Adam/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_17/kernel/m

+Adam/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_16/bias/m
{
)Adam/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_16/kernel/m

+Adam/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/m*&
_output_shapes
: *
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
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

conv2d_transpose_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_13/bias

,conv2d_transpose_13/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_13/bias*
_output_shapes
:*
dtype0

conv2d_transpose_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_13/kernel

.conv2d_transpose_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_13/kernel*&
_output_shapes
: *
dtype0

conv2d_transpose_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_12/bias

,conv2d_transpose_12/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_12/bias*
_output_shapes
: *
dtype0

conv2d_transpose_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_12/kernel

.conv2d_transpose_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_12/kernel*&
_output_shapes
: @*
dtype0

conv2d_transpose_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2d_transpose_11/bias

,conv2d_transpose_11/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_11/bias*
_output_shapes
:@*
dtype0

conv2d_transpose_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameconv2d_transpose_11/kernel

.conv2d_transpose_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_11/kernel*'
_output_shapes
:@*
dtype0

conv2d_transpose_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_10/bias

,conv2d_transpose_10/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_10/bias*
_output_shapes	
:*
dtype0

conv2d_transpose_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_10/kernel

.conv2d_transpose_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_10/kernel*(
_output_shapes
:*
dtype0
u
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_19/bias
n
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes	
:*
dtype0

conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_19/kernel

$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*(
_output_shapes
:*
dtype0
u
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_18/bias
n
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes	
:*
dtype0

conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_18/kernel
~
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*'
_output_shapes
:@*
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
:@*
dtype0

conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_17/kernel
}
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
: *
dtype0

conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
: *
dtype0

serving_default_input_9Placeholder*1
_output_shapes
:џџџџџџџџџаа*
dtype0*&
shape:џџџџџџџџџаа
К
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_9conv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_transpose_10/kernelconv2d_transpose_10/biasconv2d_transpose_11/kernelconv2d_transpose_11/biasconv2d_transpose_12/kernelconv2d_transpose_12/biasconv2d_transpose_13/kernelconv2d_transpose_13/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_6868

NoOpNoOp
Њq
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*хp
valueлpBиp Bбp
Ї
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
Й
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Й
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
z
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115*
z
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115*
* 
А
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
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

?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rate"mк#mл$mм%mн&mо'mп(mр)mс*mт+mу,mф-mх.mц/mч0mш1mщ"vъ#vы$vь%vэ&vю'vя(v№)vё*vђ+vѓ,vє-vѕ.vі/vї0vј1vљ*

Dserving_default* 
Ш
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

"kernel
#bias
 K_jit_compiled_convolution_op*
Ш
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

$kernel
%bias
 R_jit_compiled_convolution_op*
Ш
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

&kernel
'bias
 Y_jit_compiled_convolution_op*
Ш
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

(kernel
)bias
 `_jit_compiled_convolution_op*
<
"0
#1
$2
%3
&4
'5
(6
)7*
<
"0
#1
$2
%3
&4
'5
(6
)7*
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ftrace_0
gtrace_1
htrace_2
itrace_3* 
6
jtrace_0
ktrace_1
ltrace_2
mtrace_3* 
* 
Ш
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

*kernel
+bias
 t_jit_compiled_convolution_op*
Ш
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

,kernel
-bias
 {_jit_compiled_convolution_op*
Ы
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

.kernel
/bias
!_jit_compiled_convolution_op*
Я
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

0kernel
1bias
!_jit_compiled_convolution_op*
<
*0
+1
,2
-3
.4
/5
06
17*
<
*0
+1
,2
-3
.4
/5
06
17*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
PJ
VARIABLE_VALUEconv2d_16/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_16/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_17/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_17/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_18/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_18/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_19/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_19/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_10/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_10/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_transpose_11/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_11/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_transpose_12/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_12/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_transpose_13/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_13/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

"0
#1*

"0
#1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

$0
%1*

$0
%1*
* 

 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

Ѕtrace_0* 

Іtrace_0* 
* 

&0
'1*

&0
'1*
* 

Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

Ќtrace_0* 

­trace_0* 
* 

(0
)1*

(0
)1*
* 

Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

Гtrace_0* 

Дtrace_0* 
* 
* 
'
0
1
2
3
4*
* 
* 
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
*0
+1*

*0
+1*
* 

Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

Кtrace_0* 

Лtrace_0* 
* 

,0
-1*

,0
-1*
* 

Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*

Сtrace_0* 

Тtrace_0* 
* 

.0
/1*

.0
/1*
* 

Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Шtrace_0* 

Щtrace_0* 
* 

00
11*

00
11*
* 

Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Яtrace_0* 

аtrace_0* 
* 
* 
'
0
1
2
3
4*
* 
* 
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
б	variables
в	keras_api

гtotal

дcount*
M
е	variables
ж	keras_api

зtotal

иcount
й
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

г0
д1*

б	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

з0
и1*

е	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
sm
VARIABLE_VALUEAdam/conv2d_16/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_16/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_17/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_17/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_18/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_18/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_19/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_19/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/conv2d_transpose_10/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_10/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/conv2d_transpose_11/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d_transpose_11/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/conv2d_transpose_12/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d_transpose_12/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/conv2d_transpose_13/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d_transpose_13/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_16/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_16/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_17/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_17/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_18/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_18/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_19/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_19/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/conv2d_transpose_10/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_10/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/conv2d_transpose_11/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d_transpose_11/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/conv2d_transpose_12/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d_transpose_12/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE!Adam/conv2d_transpose_13/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d_transpose_13/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ћ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp.conv2d_transpose_10/kernel/Read/ReadVariableOp,conv2d_transpose_10/bias/Read/ReadVariableOp.conv2d_transpose_11/kernel/Read/ReadVariableOp,conv2d_transpose_11/bias/Read/ReadVariableOp.conv2d_transpose_12/kernel/Read/ReadVariableOp,conv2d_transpose_12/bias/Read/ReadVariableOp.conv2d_transpose_13/kernel/Read/ReadVariableOp,conv2d_transpose_13/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_16/kernel/m/Read/ReadVariableOp)Adam/conv2d_16/bias/m/Read/ReadVariableOp+Adam/conv2d_17/kernel/m/Read/ReadVariableOp)Adam/conv2d_17/bias/m/Read/ReadVariableOp+Adam/conv2d_18/kernel/m/Read/ReadVariableOp)Adam/conv2d_18/bias/m/Read/ReadVariableOp+Adam/conv2d_19/kernel/m/Read/ReadVariableOp)Adam/conv2d_19/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_10/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_10/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_11/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_11/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_12/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_12/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_13/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_13/bias/m/Read/ReadVariableOp+Adam/conv2d_16/kernel/v/Read/ReadVariableOp)Adam/conv2d_16/bias/v/Read/ReadVariableOp+Adam/conv2d_17/kernel/v/Read/ReadVariableOp)Adam/conv2d_17/bias/v/Read/ReadVariableOp+Adam/conv2d_18/kernel/v/Read/ReadVariableOp)Adam/conv2d_18/bias/v/Read/ReadVariableOp+Adam/conv2d_19/kernel/v/Read/ReadVariableOp)Adam/conv2d_19/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_10/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_10/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_11/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_11/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_12/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_12/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_13/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_13/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
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
GPU 2J 8 *&
f!R
__inference__traced_save_7928
В
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_transpose_10/kernelconv2d_transpose_10/biasconv2d_transpose_11/kernelconv2d_transpose_11/biasconv2d_transpose_12/kernelconv2d_transpose_12/biasconv2d_transpose_13/kernelconv2d_transpose_13/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_16/kernel/mAdam/conv2d_16/bias/mAdam/conv2d_17/kernel/mAdam/conv2d_17/bias/mAdam/conv2d_18/kernel/mAdam/conv2d_18/bias/mAdam/conv2d_19/kernel/mAdam/conv2d_19/bias/m!Adam/conv2d_transpose_10/kernel/mAdam/conv2d_transpose_10/bias/m!Adam/conv2d_transpose_11/kernel/mAdam/conv2d_transpose_11/bias/m!Adam/conv2d_transpose_12/kernel/mAdam/conv2d_transpose_12/bias/m!Adam/conv2d_transpose_13/kernel/mAdam/conv2d_transpose_13/bias/mAdam/conv2d_16/kernel/vAdam/conv2d_16/bias/vAdam/conv2d_17/kernel/vAdam/conv2d_17/bias/vAdam/conv2d_18/kernel/vAdam/conv2d_18/bias/vAdam/conv2d_19/kernel/vAdam/conv2d_19/bias/v!Adam/conv2d_transpose_10/kernel/vAdam/conv2d_transpose_10/bias/v!Adam/conv2d_transpose_11/kernel/vAdam/conv2d_transpose_11/bias/v!Adam/conv2d_transpose_12/kernel/vAdam/conv2d_transpose_12/bias/v!Adam/conv2d_transpose_13/kernel/vAdam/conv2d_transpose_13/bias/v*E
Tin>
<2:*
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
GPU 2J 8 *)
f$R"
 __inference__traced_restore_8109тк
У!

M__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_6287

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
: @*
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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ {
IdentityIdentityRelu:activations:0^NoOp*
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
а!

M__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_7605

inputsD
(conv2d_transpose_readvariableop_resource:.
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
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
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
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


A__inference_decoder_layer_call_and_return_conditional_losses_6497
input_104
conv2d_transpose_10_6476:'
conv2d_transpose_10_6478:	3
conv2d_transpose_11_6481:@&
conv2d_transpose_11_6483:@2
conv2d_transpose_12_6486: @&
conv2d_transpose_12_6488: 2
conv2d_transpose_13_6491: &
conv2d_transpose_13_6493:
identityЂ+conv2d_transpose_10/StatefulPartitionedCallЂ+conv2d_transpose_11/StatefulPartitionedCallЂ+conv2d_transpose_12/StatefulPartitionedCallЂ+conv2d_transpose_13/StatefulPartitionedCallЁ
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCallinput_10conv2d_transpose_10_6476conv2d_transpose_10_6478*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_6197Ь
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0conv2d_transpose_11_6481conv2d_transpose_11_6483*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_6242Ь
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0conv2d_transpose_12_6486conv2d_transpose_12_6488*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_6287Ю
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_6491conv2d_transpose_13_6493*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_6332
IdentityIdentity4conv2d_transpose_13/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџааў
NoOpNoOp,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ: : : : : : : : 2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_10
б

A__inference_encoder_layer_call_and_return_conditional_losses_6135
input_9(
conv2d_16_6114: 
conv2d_16_6116: (
conv2d_17_6119: @
conv2d_17_6121:@)
conv2d_18_6124:@
conv2d_18_6126:	*
conv2d_19_6129:
conv2d_19_6131:	
identityЂ!conv2d_16/StatefulPartitionedCallЂ!conv2d_17/StatefulPartitionedCallЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallї
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinput_9conv2d_16_6114conv2d_16_6116*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_5907
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_6119conv2d_17_6121*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_5924
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_6124conv2d_18_6126*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_5941
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0conv2d_19_6129conv2d_19_6131*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_5958
IdentityIdentity*conv2d_19/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџж
NoOpNoOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџаа: : : : : : : : 2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџаа
!
_user_specified_name	input_9

м
&__inference_model_2_layer_call_fn_6747
input_9!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	$
	unknown_9:@

unknown_10:@$

unknown_11: @

unknown_12: $

unknown_13: 

unknown_14:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *1
_output_shapes
:џџџџџџџџџаа*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_6675y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџаа: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџаа
!
_user_specified_name	input_9
Т!

M__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_7734

inputsB
(conv2d_transpose_readvariableop_resource: -
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
: *
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
от
Ѓ&
 __inference__traced_restore_8109
file_prefix;
!assignvariableop_conv2d_16_kernel: /
!assignvariableop_1_conv2d_16_bias: =
#assignvariableop_2_conv2d_17_kernel: @/
!assignvariableop_3_conv2d_17_bias:@>
#assignvariableop_4_conv2d_18_kernel:@0
!assignvariableop_5_conv2d_18_bias:	?
#assignvariableop_6_conv2d_19_kernel:0
!assignvariableop_7_conv2d_19_bias:	I
-assignvariableop_8_conv2d_transpose_10_kernel::
+assignvariableop_9_conv2d_transpose_10_bias:	I
.assignvariableop_10_conv2d_transpose_11_kernel:@:
,assignvariableop_11_conv2d_transpose_11_bias:@H
.assignvariableop_12_conv2d_transpose_12_kernel: @:
,assignvariableop_13_conv2d_transpose_12_bias: H
.assignvariableop_14_conv2d_transpose_13_kernel: :
,assignvariableop_15_conv2d_transpose_13_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: #
assignvariableop_23_total: #
assignvariableop_24_count: E
+assignvariableop_25_adam_conv2d_16_kernel_m: 7
)assignvariableop_26_adam_conv2d_16_bias_m: E
+assignvariableop_27_adam_conv2d_17_kernel_m: @7
)assignvariableop_28_adam_conv2d_17_bias_m:@F
+assignvariableop_29_adam_conv2d_18_kernel_m:@8
)assignvariableop_30_adam_conv2d_18_bias_m:	G
+assignvariableop_31_adam_conv2d_19_kernel_m:8
)assignvariableop_32_adam_conv2d_19_bias_m:	Q
5assignvariableop_33_adam_conv2d_transpose_10_kernel_m:B
3assignvariableop_34_adam_conv2d_transpose_10_bias_m:	P
5assignvariableop_35_adam_conv2d_transpose_11_kernel_m:@A
3assignvariableop_36_adam_conv2d_transpose_11_bias_m:@O
5assignvariableop_37_adam_conv2d_transpose_12_kernel_m: @A
3assignvariableop_38_adam_conv2d_transpose_12_bias_m: O
5assignvariableop_39_adam_conv2d_transpose_13_kernel_m: A
3assignvariableop_40_adam_conv2d_transpose_13_bias_m:E
+assignvariableop_41_adam_conv2d_16_kernel_v: 7
)assignvariableop_42_adam_conv2d_16_bias_v: E
+assignvariableop_43_adam_conv2d_17_kernel_v: @7
)assignvariableop_44_adam_conv2d_17_bias_v:@F
+assignvariableop_45_adam_conv2d_18_kernel_v:@8
)assignvariableop_46_adam_conv2d_18_bias_v:	G
+assignvariableop_47_adam_conv2d_19_kernel_v:8
)assignvariableop_48_adam_conv2d_19_bias_v:	Q
5assignvariableop_49_adam_conv2d_transpose_10_kernel_v:B
3assignvariableop_50_adam_conv2d_transpose_10_bias_v:	P
5assignvariableop_51_adam_conv2d_transpose_11_kernel_v:@A
3assignvariableop_52_adam_conv2d_transpose_11_bias_v:@O
5assignvariableop_53_adam_conv2d_transpose_12_kernel_v: @A
3assignvariableop_54_adam_conv2d_transpose_12_bias_v: O
5assignvariableop_55_adam_conv2d_transpose_13_kernel_v: A
3assignvariableop_56_adam_conv2d_transpose_13_bias_v:
identity_58ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9м
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
valueјBѕ:B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHх
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B У
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ў
_output_shapesы
ш::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_16_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_16_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_17_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_17_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_18_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_18_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_19_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_19_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp-assignvariableop_8_conv2d_transpose_10_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp+assignvariableop_9_conv2d_transpose_10_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp.assignvariableop_10_conv2d_transpose_11_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp,assignvariableop_11_conv2d_transpose_11_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp.assignvariableop_12_conv2d_transpose_12_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp,assignvariableop_13_conv2d_transpose_12_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp.assignvariableop_14_conv2d_transpose_13_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp,assignvariableop_15_conv2d_transpose_13_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_16_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_16_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_17_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_17_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_18_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_18_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_19_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_19_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_33AssignVariableOp5assignvariableop_33_adam_conv2d_transpose_10_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_conv2d_transpose_10_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_conv2d_transpose_11_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_conv2d_transpose_11_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_conv2d_transpose_12_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_conv2d_transpose_12_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_39AssignVariableOp5assignvariableop_39_adam_conv2d_transpose_13_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_40AssignVariableOp3assignvariableop_40_adam_conv2d_transpose_13_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_16_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_16_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_17_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_17_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_18_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_18_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_19_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_19_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_49AssignVariableOp5assignvariableop_49_adam_conv2d_transpose_10_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_50AssignVariableOp3assignvariableop_50_adam_conv2d_transpose_10_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_conv2d_transpose_11_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_conv2d_transpose_11_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_53AssignVariableOp5assignvariableop_53_adam_conv2d_transpose_12_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_54AssignVariableOp3assignvariableop_54_adam_conv2d_transpose_12_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_55AssignVariableOp5assignvariableop_55_adam_conv2d_transpose_13_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_56AssignVariableOp3assignvariableop_56_adam_conv2d_transpose_13_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Е

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: Ђ

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
р
и
"__inference_signature_wrapper_6868
input_9!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	$
	unknown_9:@

unknown_10:@$

unknown_11: @

unknown_12: $

unknown_13: 

unknown_14:
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *1
_output_shapes
:џџџџџџџџџаа*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_5889y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџаа: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџаа
!
_user_specified_name	input_9
Х
Ї
2__inference_conv2d_transpose_13_layer_call_fn_7700

inputs!
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallќ
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
GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_6332
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
Х
Ї
2__inference_conv2d_transpose_12_layer_call_fn_7657

inputs!
unknown: @
	unknown_0: 
identityЂStatefulPartitionedCallќ
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
GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_6287
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

ќ
C__inference_conv2d_17_layer_call_and_return_conditional_losses_7522

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@*
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
:џџџџџџџџџ44@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ44@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ44@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџhh : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџhh 
 
_user_specified_nameinputs

м
&__inference_model_2_layer_call_fn_6598
input_9!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	$
	unknown_9:@

unknown_10:@$

unknown_11: @

unknown_12: $

unknown_13: 

unknown_14:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *1
_output_shapes
:џџџџџџџџџаа*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_6563y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџаа: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџаа
!
_user_specified_name	input_9
У!

M__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_7691

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
: @*
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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ {
IdentityIdentityRelu:activations:0^NoOp*
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
б

A__inference_encoder_layer_call_and_return_conditional_losses_6159
input_9(
conv2d_16_6138: 
conv2d_16_6140: (
conv2d_17_6143: @
conv2d_17_6145:@)
conv2d_18_6148:@
conv2d_18_6150:	*
conv2d_19_6153:
conv2d_19_6155:	
identityЂ!conv2d_16/StatefulPartitionedCallЂ!conv2d_17/StatefulPartitionedCallЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallї
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinput_9conv2d_16_6138conv2d_16_6140*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_5907
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_6143conv2d_17_6145*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_5924
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_6148conv2d_18_6150*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_5941
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0conv2d_19_6153conv2d_19_6155*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_5958
IdentityIdentity*conv2d_19/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџж
NoOpNoOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџаа: : : : : : : : 2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџаа
!
_user_specified_name	input_9
Љ
Ю
A__inference_model_2_layer_call_and_return_conditional_losses_6785
input_9&
encoder_6750: 
encoder_6752: &
encoder_6754: @
encoder_6756:@'
encoder_6758:@
encoder_6760:	(
encoder_6762:
encoder_6764:	(
decoder_6767:
decoder_6769:	'
decoder_6771:@
decoder_6773:@&
decoder_6775: @
decoder_6777: &
decoder_6779: 
decoder_6781:
identityЂdecoder/StatefulPartitionedCallЂencoder/StatefulPartitionedCallа
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_9encoder_6750encoder_6752encoder_6754encoder_6756encoder_6758encoder_6760encoder_6762encoder_6764*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_5965ђ
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_6767decoder_6769decoder_6771decoder_6773decoder_6775decoder_6777decoder_6779decoder_6781*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_6367
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџаа: : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџаа
!
_user_specified_name	input_9


й
&__inference_decoder_layer_call_fn_7293

inputs#
unknown:
	unknown_0:	$
	unknown_1:@
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_6367y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

џ
C__inference_conv2d_19_layer_call_and_return_conditional_losses_7562

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


л
&__inference_decoder_layer_call_fn_6386
input_10#
unknown:
	unknown_0:	$
	unknown_1:@
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identityЂStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_6367y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_10


A__inference_decoder_layer_call_and_return_conditional_losses_6433

inputs4
conv2d_transpose_10_6412:'
conv2d_transpose_10_6414:	3
conv2d_transpose_11_6417:@&
conv2d_transpose_11_6419:@2
conv2d_transpose_12_6422: @&
conv2d_transpose_12_6424: 2
conv2d_transpose_13_6427: &
conv2d_transpose_13_6429:
identityЂ+conv2d_transpose_10/StatefulPartitionedCallЂ+conv2d_transpose_11/StatefulPartitionedCallЂ+conv2d_transpose_12/StatefulPartitionedCallЂ+conv2d_transpose_13/StatefulPartitionedCall
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_10_6412conv2d_transpose_10_6414*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_6197Ь
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0conv2d_transpose_11_6417conv2d_transpose_11_6419*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_6242Ь
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0conv2d_transpose_12_6422conv2d_transpose_12_6424*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_6287Ю
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_6427conv2d_transpose_13_6429*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_6332
IdentityIdentity4conv2d_transpose_13/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџааў
NoOpNoOp,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ: : : : : : : : 2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч!

M__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_7648

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
:@*
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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@{
IdentityIdentityRelu:activations:0^NoOp*
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
І
Э
A__inference_model_2_layer_call_and_return_conditional_losses_6675

inputs&
encoder_6640: 
encoder_6642: &
encoder_6644: @
encoder_6646:@'
encoder_6648:@
encoder_6650:	(
encoder_6652:
encoder_6654:	(
decoder_6657:
decoder_6659:	'
decoder_6661:@
decoder_6663:@&
decoder_6665: @
decoder_6667: &
decoder_6669: 
decoder_6671:
identityЂdecoder/StatefulPartitionedCallЂencoder/StatefulPartitionedCallЯ
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_6640encoder_6642encoder_6644encoder_6646encoder_6648encoder_6650encoder_6652encoder_6654*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_6071ђ
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_6657decoder_6659decoder_6661decoder_6663decoder_6665decoder_6667decoder_6669decoder_6671*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_6433
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџаа: : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs
ь

(__inference_conv2d_16_layer_call_fn_7491

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_5907w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџhh `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџаа: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs
q
м
A__inference_decoder_layer_call_and_return_conditional_losses_7398

inputsX
<conv2d_transpose_10_conv2d_transpose_readvariableop_resource:B
3conv2d_transpose_10_biasadd_readvariableop_resource:	W
<conv2d_transpose_11_conv2d_transpose_readvariableop_resource:@A
3conv2d_transpose_11_biasadd_readvariableop_resource:@V
<conv2d_transpose_12_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_12_biasadd_readvariableop_resource: V
<conv2d_transpose_13_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_13_biasadd_readvariableop_resource:
identityЂ*conv2d_transpose_10/BiasAdd/ReadVariableOpЂ3conv2d_transpose_10/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_11/BiasAdd/ReadVariableOpЂ3conv2d_transpose_11/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_12/BiasAdd/ReadVariableOpЂ3conv2d_transpose_12/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_13/BiasAdd/ReadVariableOpЂ3conv2d_transpose_13/conv2d_transpose/ReadVariableOpO
conv2d_transpose_10/ShapeShapeinputs*
T0*
_output_shapes
:q
'conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_10/strided_sliceStridedSlice"conv2d_transpose_10/Shape:output:00conv2d_transpose_10/strided_slice/stack:output:02conv2d_transpose_10/strided_slice/stack_1:output:02conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value
B :э
conv2d_transpose_10/stackPack*conv2d_transpose_10/strided_slice:output:0$conv2d_transpose_10/stack/1:output:0$conv2d_transpose_10/stack/2:output:0$conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_10/strided_slice_1StridedSlice"conv2d_transpose_10/stack:output:02conv2d_transpose_10/strided_slice_1/stack:output:04conv2d_transpose_10/strided_slice_1/stack_1:output:04conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskК
3conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_10_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0
$conv2d_transpose_10/conv2d_transposeConv2DBackpropInput"conv2d_transpose_10/stack:output:0;conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

*conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ф
conv2d_transpose_10/BiasAddBiasAdd-conv2d_transpose_10/conv2d_transpose:output:02conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
conv2d_transpose_10/ReluRelu$conv2d_transpose_10/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_transpose_11/ShapeShape&conv2d_transpose_10/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_11/strided_sliceStridedSlice"conv2d_transpose_11/Shape:output:00conv2d_transpose_11/strided_slice/stack:output:02conv2d_transpose_11/strided_slice/stack_1:output:02conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :4]
conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :4]
conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@э
conv2d_transpose_11/stackPack*conv2d_transpose_11/strided_slice:output:0$conv2d_transpose_11/stack/1:output:0$conv2d_transpose_11/stack/2:output:0$conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_11/strided_slice_1StridedSlice"conv2d_transpose_11/stack:output:02conv2d_transpose_11/strided_slice_1/stack:output:04conv2d_transpose_11/strided_slice_1/stack_1:output:04conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_11_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0І
$conv2d_transpose_11/conv2d_transposeConv2DBackpropInput"conv2d_transpose_11/stack:output:0;conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_10/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ44@*
paddingSAME*
strides

*conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0У
conv2d_transpose_11/BiasAddBiasAdd-conv2d_transpose_11/conv2d_transpose:output:02conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@
conv2d_transpose_11/ReluRelu$conv2d_transpose_11/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ44@o
conv2d_transpose_12/ShapeShape&conv2d_transpose_11/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_12/strided_sliceStridedSlice"conv2d_transpose_12/Shape:output:00conv2d_transpose_12/strided_slice/stack:output:02conv2d_transpose_12/strided_slice/stack_1:output:02conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :h]
conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :h]
conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B : э
conv2d_transpose_12/stackPack*conv2d_transpose_12/strided_slice:output:0$conv2d_transpose_12/stack/1:output:0$conv2d_transpose_12/stack/2:output:0$conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_12/strided_slice_1StridedSlice"conv2d_transpose_12/stack:output:02conv2d_transpose_12/strided_slice_1/stack:output:04conv2d_transpose_12/strided_slice_1/stack_1:output:04conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0І
$conv2d_transpose_12/conv2d_transposeConv2DBackpropInput"conv2d_transpose_12/stack:output:0;conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_11/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџhh *
paddingSAME*
strides

*conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0У
conv2d_transpose_12/BiasAddBiasAdd-conv2d_transpose_12/conv2d_transpose:output:02conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh 
conv2d_transpose_12/ReluRelu$conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh o
conv2d_transpose_13/ShapeShape&conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_13/strided_sliceStridedSlice"conv2d_transpose_13/Shape:output:00conv2d_transpose_13/strided_slice/stack:output:02conv2d_transpose_13/strided_slice/stack_1:output:02conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value
B :а^
conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value
B :а]
conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :э
conv2d_transpose_13/stackPack*conv2d_transpose_13/strided_slice:output:0$conv2d_transpose_13/stack/1:output:0$conv2d_transpose_13/stack/2:output:0$conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_13/strided_slice_1StridedSlice"conv2d_transpose_13/stack:output:02conv2d_transpose_13/strided_slice_1/stack:output:04conv2d_transpose_13/strided_slice_1/stack_1:output:04conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ј
$conv2d_transpose_13/conv2d_transposeConv2DBackpropInput"conv2d_transpose_13/stack:output:0;conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_12/Relu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџаа*
paddingSAME*
strides

*conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Х
conv2d_transpose_13/BiasAddBiasAdd-conv2d_transpose_13/conv2d_transpose:output:02conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџаа
conv2d_transpose_13/SigmoidSigmoid$conv2d_transpose_13/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџааx
IdentityIdentityconv2d_transpose_13/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаав
NoOpNoOp+^conv2d_transpose_10/BiasAdd/ReadVariableOp4^conv2d_transpose_10/conv2d_transpose/ReadVariableOp+^conv2d_transpose_11/BiasAdd/ReadVariableOp4^conv2d_transpose_11/conv2d_transpose/ReadVariableOp+^conv2d_transpose_12/BiasAdd/ReadVariableOp4^conv2d_transpose_12/conv2d_transpose/ReadVariableOp+^conv2d_transpose_13/BiasAdd/ReadVariableOp4^conv2d_transpose_13/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ: : : : : : : : 2X
*conv2d_transpose_10/BiasAdd/ReadVariableOp*conv2d_transpose_10/BiasAdd/ReadVariableOp2j
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp3conv2d_transpose_10/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_11/BiasAdd/ReadVariableOp*conv2d_transpose_11/BiasAdd/ReadVariableOp2j
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp3conv2d_transpose_11/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_12/BiasAdd/ReadVariableOp*conv2d_transpose_12/BiasAdd/ReadVariableOp2j
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp3conv2d_transpose_12/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_13/BiasAdd/ReadVariableOp*conv2d_transpose_13/BiasAdd/ReadVariableOp2j
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp3conv2d_transpose_13/conv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
я
 
(__inference_conv2d_19_layer_call_fn_7551

inputs#
unknown:
	unknown_0:	
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_5958x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


A__inference_decoder_layer_call_and_return_conditional_losses_6521
input_104
conv2d_transpose_10_6500:'
conv2d_transpose_10_6502:	3
conv2d_transpose_11_6505:@&
conv2d_transpose_11_6507:@2
conv2d_transpose_12_6510: @&
conv2d_transpose_12_6512: 2
conv2d_transpose_13_6515: &
conv2d_transpose_13_6517:
identityЂ+conv2d_transpose_10/StatefulPartitionedCallЂ+conv2d_transpose_11/StatefulPartitionedCallЂ+conv2d_transpose_12/StatefulPartitionedCallЂ+conv2d_transpose_13/StatefulPartitionedCallЁ
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCallinput_10conv2d_transpose_10_6500conv2d_transpose_10_6502*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_6197Ь
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0conv2d_transpose_11_6505conv2d_transpose_11_6507*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_6242Ь
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0conv2d_transpose_12_6510conv2d_transpose_12_6512*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_6287Ю
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_6515conv2d_transpose_13_6517*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_6332
IdentityIdentity4conv2d_transpose_13/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџааў
NoOpNoOp,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ: : : : : : : : 2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_10
ў'
э
A__inference_encoder_layer_call_and_return_conditional_losses_7240

inputsB
(conv2d_16_conv2d_readvariableop_resource: 7
)conv2d_16_biasadd_readvariableop_resource: B
(conv2d_17_conv2d_readvariableop_resource: @7
)conv2d_17_biasadd_readvariableop_resource:@C
(conv2d_18_conv2d_readvariableop_resource:@8
)conv2d_18_biasadd_readvariableop_resource:	D
(conv2d_19_conv2d_readvariableop_resource:8
)conv2d_19_biasadd_readvariableop_resource:	
identityЂ conv2d_16/BiasAdd/ReadVariableOpЂconv2d_16/Conv2D/ReadVariableOpЂ conv2d_17/BiasAdd/ReadVariableOpЂconv2d_17/Conv2D/ReadVariableOpЂ conv2d_18/BiasAdd/ReadVariableOpЂconv2d_18/Conv2D/ReadVariableOpЂ conv2d_19/BiasAdd/ReadVariableOpЂconv2d_19/Conv2D/ReadVariableOp
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0­
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh *
paddingSAME*
strides

 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh l
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh 
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0У
conv2d_17/Conv2DConv2Dconv2d_16/Relu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@*
paddingSAME*
strides

 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@l
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ44@
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ф
conv2d_18/Conv2DConv2Dconv2d_17/Relu:activations:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџm
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ф
conv2d_19/Conv2DConv2Dconv2d_18/Relu:activations:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџm
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџt
IdentityIdentityconv2d_19/Relu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџк
NoOpNoOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџаа: : : : : : : : 2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs
Ч!

M__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_6242

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
:@*
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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@{
IdentityIdentityRelu:activations:0^NoOp*
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
q
м
A__inference_decoder_layer_call_and_return_conditional_losses_7482

inputsX
<conv2d_transpose_10_conv2d_transpose_readvariableop_resource:B
3conv2d_transpose_10_biasadd_readvariableop_resource:	W
<conv2d_transpose_11_conv2d_transpose_readvariableop_resource:@A
3conv2d_transpose_11_biasadd_readvariableop_resource:@V
<conv2d_transpose_12_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_12_biasadd_readvariableop_resource: V
<conv2d_transpose_13_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_13_biasadd_readvariableop_resource:
identityЂ*conv2d_transpose_10/BiasAdd/ReadVariableOpЂ3conv2d_transpose_10/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_11/BiasAdd/ReadVariableOpЂ3conv2d_transpose_11/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_12/BiasAdd/ReadVariableOpЂ3conv2d_transpose_12/conv2d_transpose/ReadVariableOpЂ*conv2d_transpose_13/BiasAdd/ReadVariableOpЂ3conv2d_transpose_13/conv2d_transpose/ReadVariableOpO
conv2d_transpose_10/ShapeShapeinputs*
T0*
_output_shapes
:q
'conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_10/strided_sliceStridedSlice"conv2d_transpose_10/Shape:output:00conv2d_transpose_10/strided_slice/stack:output:02conv2d_transpose_10/strided_slice/stack_1:output:02conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value
B :э
conv2d_transpose_10/stackPack*conv2d_transpose_10/strided_slice:output:0$conv2d_transpose_10/stack/1:output:0$conv2d_transpose_10/stack/2:output:0$conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_10/strided_slice_1StridedSlice"conv2d_transpose_10/stack:output:02conv2d_transpose_10/strided_slice_1/stack:output:04conv2d_transpose_10/strided_slice_1/stack_1:output:04conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskК
3conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_10_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0
$conv2d_transpose_10/conv2d_transposeConv2DBackpropInput"conv2d_transpose_10/stack:output:0;conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

*conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ф
conv2d_transpose_10/BiasAddBiasAdd-conv2d_transpose_10/conv2d_transpose:output:02conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
conv2d_transpose_10/ReluRelu$conv2d_transpose_10/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџo
conv2d_transpose_11/ShapeShape&conv2d_transpose_10/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_11/strided_sliceStridedSlice"conv2d_transpose_11/Shape:output:00conv2d_transpose_11/strided_slice/stack:output:02conv2d_transpose_11/strided_slice/stack_1:output:02conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :4]
conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :4]
conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@э
conv2d_transpose_11/stackPack*conv2d_transpose_11/strided_slice:output:0$conv2d_transpose_11/stack/1:output:0$conv2d_transpose_11/stack/2:output:0$conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_11/strided_slice_1StridedSlice"conv2d_transpose_11/stack:output:02conv2d_transpose_11/strided_slice_1/stack:output:04conv2d_transpose_11/strided_slice_1/stack_1:output:04conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЙ
3conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_11_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0І
$conv2d_transpose_11/conv2d_transposeConv2DBackpropInput"conv2d_transpose_11/stack:output:0;conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_10/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ44@*
paddingSAME*
strides

*conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0У
conv2d_transpose_11/BiasAddBiasAdd-conv2d_transpose_11/conv2d_transpose:output:02conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@
conv2d_transpose_11/ReluRelu$conv2d_transpose_11/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ44@o
conv2d_transpose_12/ShapeShape&conv2d_transpose_11/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_12/strided_sliceStridedSlice"conv2d_transpose_12/Shape:output:00conv2d_transpose_12/strided_slice/stack:output:02conv2d_transpose_12/strided_slice/stack_1:output:02conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :h]
conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :h]
conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B : э
conv2d_transpose_12/stackPack*conv2d_transpose_12/strided_slice:output:0$conv2d_transpose_12/stack/1:output:0$conv2d_transpose_12/stack/2:output:0$conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_12/strided_slice_1StridedSlice"conv2d_transpose_12/stack:output:02conv2d_transpose_12/strided_slice_1/stack:output:04conv2d_transpose_12/strided_slice_1/stack_1:output:04conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0І
$conv2d_transpose_12/conv2d_transposeConv2DBackpropInput"conv2d_transpose_12/stack:output:0;conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_11/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџhh *
paddingSAME*
strides

*conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0У
conv2d_transpose_12/BiasAddBiasAdd-conv2d_transpose_12/conv2d_transpose:output:02conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh 
conv2d_transpose_12/ReluRelu$conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh o
conv2d_transpose_13/ShapeShape&conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!conv2d_transpose_13/strided_sliceStridedSlice"conv2d_transpose_13/Shape:output:00conv2d_transpose_13/strided_slice/stack:output:02conv2d_transpose_13/strided_slice/stack_1:output:02conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value
B :а^
conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value
B :а]
conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :э
conv2d_transpose_13/stackPack*conv2d_transpose_13/strided_slice:output:0$conv2d_transpose_13/stack/1:output:0$conv2d_transpose_13/stack/2:output:0$conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
#conv2d_transpose_13/strided_slice_1StridedSlice"conv2d_transpose_13/stack:output:02conv2d_transpose_13/strided_slice_1/stack:output:04conv2d_transpose_13/strided_slice_1/stack_1:output:04conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskИ
3conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ј
$conv2d_transpose_13/conv2d_transposeConv2DBackpropInput"conv2d_transpose_13/stack:output:0;conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_12/Relu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџаа*
paddingSAME*
strides

*conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Х
conv2d_transpose_13/BiasAddBiasAdd-conv2d_transpose_13/conv2d_transpose:output:02conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџаа
conv2d_transpose_13/SigmoidSigmoid$conv2d_transpose_13/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџааx
IdentityIdentityconv2d_transpose_13/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаав
NoOpNoOp+^conv2d_transpose_10/BiasAdd/ReadVariableOp4^conv2d_transpose_10/conv2d_transpose/ReadVariableOp+^conv2d_transpose_11/BiasAdd/ReadVariableOp4^conv2d_transpose_11/conv2d_transpose/ReadVariableOp+^conv2d_transpose_12/BiasAdd/ReadVariableOp4^conv2d_transpose_12/conv2d_transpose/ReadVariableOp+^conv2d_transpose_13/BiasAdd/ReadVariableOp4^conv2d_transpose_13/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ: : : : : : : : 2X
*conv2d_transpose_10/BiasAdd/ReadVariableOp*conv2d_transpose_10/BiasAdd/ReadVariableOp2j
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp3conv2d_transpose_10/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_11/BiasAdd/ReadVariableOp*conv2d_transpose_11/BiasAdd/ReadVariableOp2j
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp3conv2d_transpose_11/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_12/BiasAdd/ReadVariableOp*conv2d_transpose_12/BiasAdd/ReadVariableOp2j
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp3conv2d_transpose_12/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_13/BiasAdd/ReadVariableOp*conv2d_transpose_13/BiasAdd/ReadVariableOp2j
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp3conv2d_transpose_13/conv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ќ
C__inference_conv2d_16_layer_call_and_return_conditional_losses_5907

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџhh w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџаа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs

џ
C__inference_conv2d_19_layer_call_and_return_conditional_losses_5958

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў'
э
A__inference_encoder_layer_call_and_return_conditional_losses_7272

inputsB
(conv2d_16_conv2d_readvariableop_resource: 7
)conv2d_16_biasadd_readvariableop_resource: B
(conv2d_17_conv2d_readvariableop_resource: @7
)conv2d_17_biasadd_readvariableop_resource:@C
(conv2d_18_conv2d_readvariableop_resource:@8
)conv2d_18_biasadd_readvariableop_resource:	D
(conv2d_19_conv2d_readvariableop_resource:8
)conv2d_19_biasadd_readvariableop_resource:	
identityЂ conv2d_16/BiasAdd/ReadVariableOpЂconv2d_16/Conv2D/ReadVariableOpЂ conv2d_17/BiasAdd/ReadVariableOpЂconv2d_17/Conv2D/ReadVariableOpЂ conv2d_18/BiasAdd/ReadVariableOpЂconv2d_18/Conv2D/ReadVariableOpЂ conv2d_19/BiasAdd/ReadVariableOpЂconv2d_19/Conv2D/ReadVariableOp
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0­
conv2d_16/Conv2DConv2Dinputs'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh *
paddingSAME*
strides

 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh l
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh 
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0У
conv2d_17/Conv2DConv2Dconv2d_16/Relu:activations:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@*
paddingSAME*
strides

 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@l
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ44@
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ф
conv2d_18/Conv2DConv2Dconv2d_17/Relu:activations:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџm
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ф
conv2d_19/Conv2DConv2Dconv2d_18/Relu:activations:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџm
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџt
IdentityIdentityconv2d_19/Relu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџк
NoOpNoOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџаа: : : : : : : : 2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs


A__inference_decoder_layer_call_and_return_conditional_losses_6367

inputs4
conv2d_transpose_10_6346:'
conv2d_transpose_10_6348:	3
conv2d_transpose_11_6351:@&
conv2d_transpose_11_6353:@2
conv2d_transpose_12_6356: @&
conv2d_transpose_12_6358: 2
conv2d_transpose_13_6361: &
conv2d_transpose_13_6363:
identityЂ+conv2d_transpose_10/StatefulPartitionedCallЂ+conv2d_transpose_11/StatefulPartitionedCallЂ+conv2d_transpose_12/StatefulPartitionedCallЂ+conv2d_transpose_13/StatefulPartitionedCall
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_10_6346conv2d_transpose_10_6348*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_6197Ь
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0conv2d_transpose_11_6351conv2d_transpose_11_6353*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_6242Ь
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0conv2d_transpose_12_6356conv2d_transpose_12_6358*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_6287Ю
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_6361conv2d_transpose_13_6363*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_6332
IdentityIdentity4conv2d_transpose_13/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџааў
NoOpNoOp,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ: : : : : : : : 2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї
щ
A__inference_model_2_layer_call_and_return_conditional_losses_7166

inputsJ
0encoder_conv2d_16_conv2d_readvariableop_resource: ?
1encoder_conv2d_16_biasadd_readvariableop_resource: J
0encoder_conv2d_17_conv2d_readvariableop_resource: @?
1encoder_conv2d_17_biasadd_readvariableop_resource:@K
0encoder_conv2d_18_conv2d_readvariableop_resource:@@
1encoder_conv2d_18_biasadd_readvariableop_resource:	L
0encoder_conv2d_19_conv2d_readvariableop_resource:@
1encoder_conv2d_19_biasadd_readvariableop_resource:	`
Ddecoder_conv2d_transpose_10_conv2d_transpose_readvariableop_resource:J
;decoder_conv2d_transpose_10_biasadd_readvariableop_resource:	_
Ddecoder_conv2d_transpose_11_conv2d_transpose_readvariableop_resource:@I
;decoder_conv2d_transpose_11_biasadd_readvariableop_resource:@^
Ddecoder_conv2d_transpose_12_conv2d_transpose_readvariableop_resource: @I
;decoder_conv2d_transpose_12_biasadd_readvariableop_resource: ^
Ddecoder_conv2d_transpose_13_conv2d_transpose_readvariableop_resource: I
;decoder_conv2d_transpose_13_biasadd_readvariableop_resource:
identityЂ2decoder/conv2d_transpose_10/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_11/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_12/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_13/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOpЂ(encoder/conv2d_16/BiasAdd/ReadVariableOpЂ'encoder/conv2d_16/Conv2D/ReadVariableOpЂ(encoder/conv2d_17/BiasAdd/ReadVariableOpЂ'encoder/conv2d_17/Conv2D/ReadVariableOpЂ(encoder/conv2d_18/BiasAdd/ReadVariableOpЂ'encoder/conv2d_18/Conv2D/ReadVariableOpЂ(encoder/conv2d_19/BiasAdd/ReadVariableOpЂ'encoder/conv2d_19/Conv2D/ReadVariableOp 
'encoder/conv2d_16/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Н
encoder/conv2d_16/Conv2DConv2Dinputs/encoder/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh *
paddingSAME*
strides

(encoder/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Г
encoder/conv2d_16/BiasAddBiasAdd!encoder/conv2d_16/Conv2D:output:00encoder/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh |
encoder/conv2d_16/ReluRelu"encoder/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh  
'encoder/conv2d_17/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0л
encoder/conv2d_17/Conv2DConv2D$encoder/conv2d_16/Relu:activations:0/encoder/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@*
paddingSAME*
strides

(encoder/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
encoder/conv2d_17/BiasAddBiasAdd!encoder/conv2d_17/Conv2D:output:00encoder/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@|
encoder/conv2d_17/ReluRelu"encoder/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ44@Ё
'encoder/conv2d_18/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_18_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0м
encoder/conv2d_18/Conv2DConv2D$encoder/conv2d_17/Relu:activations:0/encoder/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

(encoder/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Д
encoder/conv2d_18/BiasAddBiasAdd!encoder/conv2d_18/Conv2D:output:00encoder/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ}
encoder/conv2d_18/ReluRelu"encoder/conv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЂ
'encoder/conv2d_19/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0м
encoder/conv2d_19/Conv2DConv2D$encoder/conv2d_18/Relu:activations:0/encoder/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

(encoder/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Д
encoder/conv2d_19/BiasAddBiasAdd!encoder/conv2d_19/Conv2D:output:00encoder/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ}
encoder/conv2d_19/ReluRelu"encoder/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџu
!decoder/conv2d_transpose_10/ShapeShape$encoder/conv2d_19/Relu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_10/strided_sliceStridedSlice*decoder/conv2d_transpose_10/Shape:output:08decoder/conv2d_transpose_10/strided_slice/stack:output:0:decoder/conv2d_transpose_10/strided_slice/stack_1:output:0:decoder/conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :f
#decoder/conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value
B :
!decoder/conv2d_transpose_10/stackPack2decoder/conv2d_transpose_10/strided_slice:output:0,decoder/conv2d_transpose_10/stack/1:output:0,decoder/conv2d_transpose_10/stack/2:output:0,decoder/conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_10/strided_slice_1StridedSlice*decoder/conv2d_transpose_10/stack:output:0:decoder/conv2d_transpose_10/strided_slice_1/stack:output:0<decoder/conv2d_transpose_10/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЪ
;decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_10_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Н
,decoder/conv2d_transpose_10/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_10/stack:output:0Cdecoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0$encoder/conv2d_19/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Ћ
2decoder/conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0м
#decoder/conv2d_transpose_10/BiasAddBiasAdd5decoder/conv2d_transpose_10/conv2d_transpose:output:0:decoder/conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
 decoder/conv2d_transpose_10/ReluRelu,decoder/conv2d_transpose_10/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
!decoder/conv2d_transpose_11/ShapeShape.decoder/conv2d_transpose_10/Relu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_11/strided_sliceStridedSlice*decoder/conv2d_transpose_11/Shape:output:08decoder/conv2d_transpose_11/strided_slice/stack:output:0:decoder/conv2d_transpose_11/strided_slice/stack_1:output:0:decoder/conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :4e
#decoder/conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :4e
#decoder/conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@
!decoder/conv2d_transpose_11/stackPack2decoder/conv2d_transpose_11/strided_slice:output:0,decoder/conv2d_transpose_11/stack/1:output:0,decoder/conv2d_transpose_11/stack/2:output:0,decoder/conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_11/strided_slice_1StridedSlice*decoder/conv2d_transpose_11/stack:output:0:decoder/conv2d_transpose_11/strided_slice_1/stack:output:0<decoder/conv2d_transpose_11/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
;decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_11_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0Ц
,decoder/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_11/stack:output:0Cdecoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_10/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ44@*
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0л
#decoder/conv2d_transpose_11/BiasAddBiasAdd5decoder/conv2d_transpose_11/conv2d_transpose:output:0:decoder/conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@
 decoder/conv2d_transpose_11/ReluRelu,decoder/conv2d_transpose_11/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ44@
!decoder/conv2d_transpose_12/ShapeShape.decoder/conv2d_transpose_11/Relu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_12/strided_sliceStridedSlice*decoder/conv2d_transpose_12/Shape:output:08decoder/conv2d_transpose_12/strided_slice/stack:output:0:decoder/conv2d_transpose_12/strided_slice/stack_1:output:0:decoder/conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :he
#decoder/conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :he
#decoder/conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
!decoder/conv2d_transpose_12/stackPack2decoder/conv2d_transpose_12/strided_slice:output:0,decoder/conv2d_transpose_12/stack/1:output:0,decoder/conv2d_transpose_12/stack/2:output:0,decoder/conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_12/strided_slice_1StridedSlice*decoder/conv2d_transpose_12/stack:output:0:decoder/conv2d_transpose_12/strided_slice_1/stack:output:0<decoder/conv2d_transpose_12/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
;decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0Ц
,decoder/conv2d_transpose_12/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_12/stack:output:0Cdecoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_11/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџhh *
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0л
#decoder/conv2d_transpose_12/BiasAddBiasAdd5decoder/conv2d_transpose_12/conv2d_transpose:output:0:decoder/conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh 
 decoder/conv2d_transpose_12/ReluRelu,decoder/conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh 
!decoder/conv2d_transpose_13/ShapeShape.decoder/conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_13/strided_sliceStridedSlice*decoder/conv2d_transpose_13/Shape:output:08decoder/conv2d_transpose_13/strided_slice/stack:output:0:decoder/conv2d_transpose_13/strided_slice/stack_1:output:0:decoder/conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#decoder/conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value
B :аf
#decoder/conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value
B :аe
#decoder/conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
!decoder/conv2d_transpose_13/stackPack2decoder/conv2d_transpose_13/strided_slice:output:0,decoder/conv2d_transpose_13/stack/1:output:0,decoder/conv2d_transpose_13/stack/2:output:0,decoder/conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_13/strided_slice_1StridedSlice*decoder/conv2d_transpose_13/stack:output:0:decoder/conv2d_transpose_13/strided_slice_1/stack:output:0<decoder/conv2d_transpose_13/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
;decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ш
,decoder/conv2d_transpose_13/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_13/stack:output:0Cdecoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_12/Relu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџаа*
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0н
#decoder/conv2d_transpose_13/BiasAddBiasAdd5decoder/conv2d_transpose_13/conv2d_transpose:output:0:decoder/conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџаа
#decoder/conv2d_transpose_13/SigmoidSigmoid,decoder/conv2d_transpose_13/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџаа
IdentityIdentity'decoder/conv2d_transpose_13/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаац
NoOpNoOp3^decoder/conv2d_transpose_10/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_11/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_12/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_13/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOp)^encoder/conv2d_16/BiasAdd/ReadVariableOp(^encoder/conv2d_16/Conv2D/ReadVariableOp)^encoder/conv2d_17/BiasAdd/ReadVariableOp(^encoder/conv2d_17/Conv2D/ReadVariableOp)^encoder/conv2d_18/BiasAdd/ReadVariableOp(^encoder/conv2d_18/Conv2D/ReadVariableOp)^encoder/conv2d_19/BiasAdd/ReadVariableOp(^encoder/conv2d_19/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџаа: : : : : : : : : : : : : : : : 2h
2decoder/conv2d_transpose_10/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_10/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_11/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_11/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_12/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_12/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_13/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_13/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOp2T
(encoder/conv2d_16/BiasAdd/ReadVariableOp(encoder/conv2d_16/BiasAdd/ReadVariableOp2R
'encoder/conv2d_16/Conv2D/ReadVariableOp'encoder/conv2d_16/Conv2D/ReadVariableOp2T
(encoder/conv2d_17/BiasAdd/ReadVariableOp(encoder/conv2d_17/BiasAdd/ReadVariableOp2R
'encoder/conv2d_17/Conv2D/ReadVariableOp'encoder/conv2d_17/Conv2D/ReadVariableOp2T
(encoder/conv2d_18/BiasAdd/ReadVariableOp(encoder/conv2d_18/BiasAdd/ReadVariableOp2R
'encoder/conv2d_18/Conv2D/ReadVariableOp'encoder/conv2d_18/Conv2D/ReadVariableOp2T
(encoder/conv2d_19/BiasAdd/ReadVariableOp(encoder/conv2d_19/BiasAdd/ReadVariableOp2R
'encoder/conv2d_19/Conv2D/ReadVariableOp'encoder/conv2d_19/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs


к
&__inference_encoder_layer_call_fn_7187

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	
identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_5965x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџаа: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs


л
&__inference_encoder_layer_call_fn_6111
input_9!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_6071x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџаа: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџаа
!
_user_specified_name	input_9
Ь
Њ
2__inference_conv2d_transpose_10_layer_call_fn_7571

inputs#
unknown:
	unknown_0:	
identityЂStatefulPartitionedCall§
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
GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_6197
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ь

(__inference_conv2d_18_layer_call_fn_7531

inputs"
unknown:@
	unknown_0:	
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_5941x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ44@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ44@
 
_user_specified_nameinputs

ў
C__inference_conv2d_18_layer_call_and_return_conditional_losses_5941

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ44@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ44@
 
_user_specified_nameinputs

ќ
C__inference_conv2d_16_layer_call_and_return_conditional_losses_7502

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџhh w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџаа: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs
Т!

M__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_6332

inputsB
(conv2d_transpose_readvariableop_resource: -
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
: *
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
ш

(__inference_conv2d_17_layer_call_fn_7511

inputs!
unknown: @
	unknown_0:@
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_5924w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ44@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџhh : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџhh 
 
_user_specified_nameinputs
Ю

A__inference_encoder_layer_call_and_return_conditional_losses_6071

inputs(
conv2d_16_6050: 
conv2d_16_6052: (
conv2d_17_6055: @
conv2d_17_6057:@)
conv2d_18_6060:@
conv2d_18_6062:	*
conv2d_19_6065:
conv2d_19_6067:	
identityЂ!conv2d_16/StatefulPartitionedCallЂ!conv2d_17/StatefulPartitionedCallЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallі
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_16_6050conv2d_16_6052*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_5907
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_6055conv2d_17_6057*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_5924
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_6060conv2d_18_6062*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_5941
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0conv2d_19_6065conv2d_19_6067*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_5958
IdentityIdentity*conv2d_19/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџж
NoOpNoOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџаа: : : : : : : : 2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs
њИ
Ш
__inference__wrapped_model_5889
input_9R
8model_2_encoder_conv2d_16_conv2d_readvariableop_resource: G
9model_2_encoder_conv2d_16_biasadd_readvariableop_resource: R
8model_2_encoder_conv2d_17_conv2d_readvariableop_resource: @G
9model_2_encoder_conv2d_17_biasadd_readvariableop_resource:@S
8model_2_encoder_conv2d_18_conv2d_readvariableop_resource:@H
9model_2_encoder_conv2d_18_biasadd_readvariableop_resource:	T
8model_2_encoder_conv2d_19_conv2d_readvariableop_resource:H
9model_2_encoder_conv2d_19_biasadd_readvariableop_resource:	h
Lmodel_2_decoder_conv2d_transpose_10_conv2d_transpose_readvariableop_resource:R
Cmodel_2_decoder_conv2d_transpose_10_biasadd_readvariableop_resource:	g
Lmodel_2_decoder_conv2d_transpose_11_conv2d_transpose_readvariableop_resource:@Q
Cmodel_2_decoder_conv2d_transpose_11_biasadd_readvariableop_resource:@f
Lmodel_2_decoder_conv2d_transpose_12_conv2d_transpose_readvariableop_resource: @Q
Cmodel_2_decoder_conv2d_transpose_12_biasadd_readvariableop_resource: f
Lmodel_2_decoder_conv2d_transpose_13_conv2d_transpose_readvariableop_resource: Q
Cmodel_2_decoder_conv2d_transpose_13_biasadd_readvariableop_resource:
identityЂ:model_2/decoder/conv2d_transpose_10/BiasAdd/ReadVariableOpЂCmodel_2/decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOpЂ:model_2/decoder/conv2d_transpose_11/BiasAdd/ReadVariableOpЂCmodel_2/decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOpЂ:model_2/decoder/conv2d_transpose_12/BiasAdd/ReadVariableOpЂCmodel_2/decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOpЂ:model_2/decoder/conv2d_transpose_13/BiasAdd/ReadVariableOpЂCmodel_2/decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOpЂ0model_2/encoder/conv2d_16/BiasAdd/ReadVariableOpЂ/model_2/encoder/conv2d_16/Conv2D/ReadVariableOpЂ0model_2/encoder/conv2d_17/BiasAdd/ReadVariableOpЂ/model_2/encoder/conv2d_17/Conv2D/ReadVariableOpЂ0model_2/encoder/conv2d_18/BiasAdd/ReadVariableOpЂ/model_2/encoder/conv2d_18/Conv2D/ReadVariableOpЂ0model_2/encoder/conv2d_19/BiasAdd/ReadVariableOpЂ/model_2/encoder/conv2d_19/Conv2D/ReadVariableOpА
/model_2/encoder/conv2d_16/Conv2D/ReadVariableOpReadVariableOp8model_2_encoder_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ю
 model_2/encoder/conv2d_16/Conv2DConv2Dinput_97model_2/encoder/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh *
paddingSAME*
strides
І
0model_2/encoder/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp9model_2_encoder_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ы
!model_2/encoder/conv2d_16/BiasAddBiasAdd)model_2/encoder/conv2d_16/Conv2D:output:08model_2/encoder/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh 
model_2/encoder/conv2d_16/ReluRelu*model_2/encoder/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh А
/model_2/encoder/conv2d_17/Conv2D/ReadVariableOpReadVariableOp8model_2_encoder_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ѓ
 model_2/encoder/conv2d_17/Conv2DConv2D,model_2/encoder/conv2d_16/Relu:activations:07model_2/encoder/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@*
paddingSAME*
strides
І
0model_2/encoder/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp9model_2_encoder_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ы
!model_2/encoder/conv2d_17/BiasAddBiasAdd)model_2/encoder/conv2d_17/Conv2D:output:08model_2/encoder/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@
model_2/encoder/conv2d_17/ReluRelu*model_2/encoder/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ44@Б
/model_2/encoder/conv2d_18/Conv2D/ReadVariableOpReadVariableOp8model_2_encoder_conv2d_18_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0є
 model_2/encoder/conv2d_18/Conv2DConv2D,model_2/encoder/conv2d_17/Relu:activations:07model_2/encoder/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Ї
0model_2/encoder/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp9model_2_encoder_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ь
!model_2/encoder/conv2d_18/BiasAddBiasAdd)model_2/encoder/conv2d_18/Conv2D:output:08model_2/encoder/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
model_2/encoder/conv2d_18/ReluRelu*model_2/encoder/conv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџВ
/model_2/encoder/conv2d_19/Conv2D/ReadVariableOpReadVariableOp8model_2_encoder_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0є
 model_2/encoder/conv2d_19/Conv2DConv2D,model_2/encoder/conv2d_18/Relu:activations:07model_2/encoder/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Ї
0model_2/encoder/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp9model_2_encoder_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ь
!model_2/encoder/conv2d_19/BiasAddBiasAdd)model_2/encoder/conv2d_19/Conv2D:output:08model_2/encoder/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
model_2/encoder/conv2d_19/ReluRelu*model_2/encoder/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
)model_2/decoder/conv2d_transpose_10/ShapeShape,model_2/encoder/conv2d_19/Relu:activations:0*
T0*
_output_shapes
:
7model_2/decoder/conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9model_2/decoder/conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9model_2/decoder/conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1model_2/decoder/conv2d_transpose_10/strided_sliceStridedSlice2model_2/decoder/conv2d_transpose_10/Shape:output:0@model_2/decoder/conv2d_transpose_10/strided_slice/stack:output:0Bmodel_2/decoder/conv2d_transpose_10/strided_slice/stack_1:output:0Bmodel_2/decoder/conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+model_2/decoder/conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :m
+model_2/decoder/conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :n
+model_2/decoder/conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value
B :Н
)model_2/decoder/conv2d_transpose_10/stackPack:model_2/decoder/conv2d_transpose_10/strided_slice:output:04model_2/decoder/conv2d_transpose_10/stack/1:output:04model_2/decoder/conv2d_transpose_10/stack/2:output:04model_2/decoder/conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:
9model_2/decoder/conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;model_2/decoder/conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;model_2/decoder/conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3model_2/decoder/conv2d_transpose_10/strided_slice_1StridedSlice2model_2/decoder/conv2d_transpose_10/stack:output:0Bmodel_2/decoder/conv2d_transpose_10/strided_slice_1/stack:output:0Dmodel_2/decoder/conv2d_transpose_10/strided_slice_1/stack_1:output:0Dmodel_2/decoder/conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskк
Cmodel_2/decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOpLmodel_2_decoder_conv2d_transpose_10_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0н
4model_2/decoder/conv2d_transpose_10/conv2d_transposeConv2DBackpropInput2model_2/decoder/conv2d_transpose_10/stack:output:0Kmodel_2/decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0,model_2/encoder/conv2d_19/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Л
:model_2/decoder/conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOpCmodel_2_decoder_conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0є
+model_2/decoder/conv2d_transpose_10/BiasAddBiasAdd=model_2/decoder/conv2d_transpose_10/conv2d_transpose:output:0Bmodel_2/decoder/conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЁ
(model_2/decoder/conv2d_transpose_10/ReluRelu4model_2/decoder/conv2d_transpose_10/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
)model_2/decoder/conv2d_transpose_11/ShapeShape6model_2/decoder/conv2d_transpose_10/Relu:activations:0*
T0*
_output_shapes
:
7model_2/decoder/conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9model_2/decoder/conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9model_2/decoder/conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1model_2/decoder/conv2d_transpose_11/strided_sliceStridedSlice2model_2/decoder/conv2d_transpose_11/Shape:output:0@model_2/decoder/conv2d_transpose_11/strided_slice/stack:output:0Bmodel_2/decoder/conv2d_transpose_11/strided_slice/stack_1:output:0Bmodel_2/decoder/conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+model_2/decoder/conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :4m
+model_2/decoder/conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :4m
+model_2/decoder/conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@Н
)model_2/decoder/conv2d_transpose_11/stackPack:model_2/decoder/conv2d_transpose_11/strided_slice:output:04model_2/decoder/conv2d_transpose_11/stack/1:output:04model_2/decoder/conv2d_transpose_11/stack/2:output:04model_2/decoder/conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:
9model_2/decoder/conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;model_2/decoder/conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;model_2/decoder/conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3model_2/decoder/conv2d_transpose_11/strided_slice_1StridedSlice2model_2/decoder/conv2d_transpose_11/stack:output:0Bmodel_2/decoder/conv2d_transpose_11/strided_slice_1/stack:output:0Dmodel_2/decoder/conv2d_transpose_11/strided_slice_1/stack_1:output:0Dmodel_2/decoder/conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskй
Cmodel_2/decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOpLmodel_2_decoder_conv2d_transpose_11_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0ц
4model_2/decoder/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput2model_2/decoder/conv2d_transpose_11/stack:output:0Kmodel_2/decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:06model_2/decoder/conv2d_transpose_10/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ44@*
paddingSAME*
strides
К
:model_2/decoder/conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOpCmodel_2_decoder_conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ѓ
+model_2/decoder/conv2d_transpose_11/BiasAddBiasAdd=model_2/decoder/conv2d_transpose_11/conv2d_transpose:output:0Bmodel_2/decoder/conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@ 
(model_2/decoder/conv2d_transpose_11/ReluRelu4model_2/decoder/conv2d_transpose_11/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ44@
)model_2/decoder/conv2d_transpose_12/ShapeShape6model_2/decoder/conv2d_transpose_11/Relu:activations:0*
T0*
_output_shapes
:
7model_2/decoder/conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9model_2/decoder/conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9model_2/decoder/conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1model_2/decoder/conv2d_transpose_12/strided_sliceStridedSlice2model_2/decoder/conv2d_transpose_12/Shape:output:0@model_2/decoder/conv2d_transpose_12/strided_slice/stack:output:0Bmodel_2/decoder/conv2d_transpose_12/strided_slice/stack_1:output:0Bmodel_2/decoder/conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+model_2/decoder/conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :hm
+model_2/decoder/conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :hm
+model_2/decoder/conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B : Н
)model_2/decoder/conv2d_transpose_12/stackPack:model_2/decoder/conv2d_transpose_12/strided_slice:output:04model_2/decoder/conv2d_transpose_12/stack/1:output:04model_2/decoder/conv2d_transpose_12/stack/2:output:04model_2/decoder/conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:
9model_2/decoder/conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;model_2/decoder/conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;model_2/decoder/conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3model_2/decoder/conv2d_transpose_12/strided_slice_1StridedSlice2model_2/decoder/conv2d_transpose_12/stack:output:0Bmodel_2/decoder/conv2d_transpose_12/strided_slice_1/stack:output:0Dmodel_2/decoder/conv2d_transpose_12/strided_slice_1/stack_1:output:0Dmodel_2/decoder/conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
Cmodel_2/decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOpLmodel_2_decoder_conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0ц
4model_2/decoder/conv2d_transpose_12/conv2d_transposeConv2DBackpropInput2model_2/decoder/conv2d_transpose_12/stack:output:0Kmodel_2/decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:06model_2/decoder/conv2d_transpose_11/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџhh *
paddingSAME*
strides
К
:model_2/decoder/conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOpCmodel_2_decoder_conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ѓ
+model_2/decoder/conv2d_transpose_12/BiasAddBiasAdd=model_2/decoder/conv2d_transpose_12/conv2d_transpose:output:0Bmodel_2/decoder/conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh  
(model_2/decoder/conv2d_transpose_12/ReluRelu4model_2/decoder/conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh 
)model_2/decoder/conv2d_transpose_13/ShapeShape6model_2/decoder/conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:
7model_2/decoder/conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9model_2/decoder/conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9model_2/decoder/conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1model_2/decoder/conv2d_transpose_13/strided_sliceStridedSlice2model_2/decoder/conv2d_transpose_13/Shape:output:0@model_2/decoder/conv2d_transpose_13/strided_slice/stack:output:0Bmodel_2/decoder/conv2d_transpose_13/strided_slice/stack_1:output:0Bmodel_2/decoder/conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
+model_2/decoder/conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value
B :аn
+model_2/decoder/conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value
B :аm
+model_2/decoder/conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Н
)model_2/decoder/conv2d_transpose_13/stackPack:model_2/decoder/conv2d_transpose_13/strided_slice:output:04model_2/decoder/conv2d_transpose_13/stack/1:output:04model_2/decoder/conv2d_transpose_13/stack/2:output:04model_2/decoder/conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:
9model_2/decoder/conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;model_2/decoder/conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;model_2/decoder/conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3model_2/decoder/conv2d_transpose_13/strided_slice_1StridedSlice2model_2/decoder/conv2d_transpose_13/stack:output:0Bmodel_2/decoder/conv2d_transpose_13/strided_slice_1/stack:output:0Dmodel_2/decoder/conv2d_transpose_13/strided_slice_1/stack_1:output:0Dmodel_2/decoder/conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
Cmodel_2/decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOpLmodel_2_decoder_conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0ш
4model_2/decoder/conv2d_transpose_13/conv2d_transposeConv2DBackpropInput2model_2/decoder/conv2d_transpose_13/stack:output:0Kmodel_2/decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:06model_2/decoder/conv2d_transpose_12/Relu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџаа*
paddingSAME*
strides
К
:model_2/decoder/conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOpCmodel_2_decoder_conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
+model_2/decoder/conv2d_transpose_13/BiasAddBiasAdd=model_2/decoder/conv2d_transpose_13/conv2d_transpose:output:0Bmodel_2/decoder/conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџааЈ
+model_2/decoder/conv2d_transpose_13/SigmoidSigmoid4model_2/decoder/conv2d_transpose_13/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџаа
IdentityIdentity/model_2/decoder/conv2d_transpose_13/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаац
NoOpNoOp;^model_2/decoder/conv2d_transpose_10/BiasAdd/ReadVariableOpD^model_2/decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOp;^model_2/decoder/conv2d_transpose_11/BiasAdd/ReadVariableOpD^model_2/decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOp;^model_2/decoder/conv2d_transpose_12/BiasAdd/ReadVariableOpD^model_2/decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOp;^model_2/decoder/conv2d_transpose_13/BiasAdd/ReadVariableOpD^model_2/decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOp1^model_2/encoder/conv2d_16/BiasAdd/ReadVariableOp0^model_2/encoder/conv2d_16/Conv2D/ReadVariableOp1^model_2/encoder/conv2d_17/BiasAdd/ReadVariableOp0^model_2/encoder/conv2d_17/Conv2D/ReadVariableOp1^model_2/encoder/conv2d_18/BiasAdd/ReadVariableOp0^model_2/encoder/conv2d_18/Conv2D/ReadVariableOp1^model_2/encoder/conv2d_19/BiasAdd/ReadVariableOp0^model_2/encoder/conv2d_19/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџаа: : : : : : : : : : : : : : : : 2x
:model_2/decoder/conv2d_transpose_10/BiasAdd/ReadVariableOp:model_2/decoder/conv2d_transpose_10/BiasAdd/ReadVariableOp2
Cmodel_2/decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOpCmodel_2/decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOp2x
:model_2/decoder/conv2d_transpose_11/BiasAdd/ReadVariableOp:model_2/decoder/conv2d_transpose_11/BiasAdd/ReadVariableOp2
Cmodel_2/decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOpCmodel_2/decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOp2x
:model_2/decoder/conv2d_transpose_12/BiasAdd/ReadVariableOp:model_2/decoder/conv2d_transpose_12/BiasAdd/ReadVariableOp2
Cmodel_2/decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOpCmodel_2/decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOp2x
:model_2/decoder/conv2d_transpose_13/BiasAdd/ReadVariableOp:model_2/decoder/conv2d_transpose_13/BiasAdd/ReadVariableOp2
Cmodel_2/decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOpCmodel_2/decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOp2d
0model_2/encoder/conv2d_16/BiasAdd/ReadVariableOp0model_2/encoder/conv2d_16/BiasAdd/ReadVariableOp2b
/model_2/encoder/conv2d_16/Conv2D/ReadVariableOp/model_2/encoder/conv2d_16/Conv2D/ReadVariableOp2d
0model_2/encoder/conv2d_17/BiasAdd/ReadVariableOp0model_2/encoder/conv2d_17/BiasAdd/ReadVariableOp2b
/model_2/encoder/conv2d_17/Conv2D/ReadVariableOp/model_2/encoder/conv2d_17/Conv2D/ReadVariableOp2d
0model_2/encoder/conv2d_18/BiasAdd/ReadVariableOp0model_2/encoder/conv2d_18/BiasAdd/ReadVariableOp2b
/model_2/encoder/conv2d_18/Conv2D/ReadVariableOp/model_2/encoder/conv2d_18/Conv2D/ReadVariableOp2d
0model_2/encoder/conv2d_19/BiasAdd/ReadVariableOp0model_2/encoder/conv2d_19/BiasAdd/ReadVariableOp2b
/model_2/encoder/conv2d_19/Conv2D/ReadVariableOp/model_2/encoder/conv2d_19/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:џџџџџџџџџаа
!
_user_specified_name	input_9
а!

M__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_6197

inputsD
(conv2d_transpose_readvariableop_resource:.
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
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
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
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџk
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ|
IdentityIdentityRelu:activations:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
йq
Ж
__inference__traced_save_7928
file_prefix/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop9
5savev2_conv2d_transpose_10_kernel_read_readvariableop7
3savev2_conv2d_transpose_10_bias_read_readvariableop9
5savev2_conv2d_transpose_11_kernel_read_readvariableop7
3savev2_conv2d_transpose_11_bias_read_readvariableop9
5savev2_conv2d_transpose_12_kernel_read_readvariableop7
3savev2_conv2d_transpose_12_bias_read_readvariableop9
5savev2_conv2d_transpose_13_kernel_read_readvariableop7
3savev2_conv2d_transpose_13_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_16_kernel_m_read_readvariableop4
0savev2_adam_conv2d_16_bias_m_read_readvariableop6
2savev2_adam_conv2d_17_kernel_m_read_readvariableop4
0savev2_adam_conv2d_17_bias_m_read_readvariableop6
2savev2_adam_conv2d_18_kernel_m_read_readvariableop4
0savev2_adam_conv2d_18_bias_m_read_readvariableop6
2savev2_adam_conv2d_19_kernel_m_read_readvariableop4
0savev2_adam_conv2d_19_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_10_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_10_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_11_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_11_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_12_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_12_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_13_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_13_bias_m_read_readvariableop6
2savev2_adam_conv2d_16_kernel_v_read_readvariableop4
0savev2_adam_conv2d_16_bias_v_read_readvariableop6
2savev2_adam_conv2d_17_kernel_v_read_readvariableop4
0savev2_adam_conv2d_17_bias_v_read_readvariableop6
2savev2_adam_conv2d_18_kernel_v_read_readvariableop4
0savev2_adam_conv2d_18_bias_v_read_readvariableop6
2savev2_adam_conv2d_19_kernel_v_read_readvariableop4
0savev2_adam_conv2d_19_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_10_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_10_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_11_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_11_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_12_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_12_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_13_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_13_bias_v_read_readvariableop
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
: й
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
valueјBѕ:B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHт
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B б
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop5savev2_conv2d_transpose_10_kernel_read_readvariableop3savev2_conv2d_transpose_10_bias_read_readvariableop5savev2_conv2d_transpose_11_kernel_read_readvariableop3savev2_conv2d_transpose_11_bias_read_readvariableop5savev2_conv2d_transpose_12_kernel_read_readvariableop3savev2_conv2d_transpose_12_bias_read_readvariableop5savev2_conv2d_transpose_13_kernel_read_readvariableop3savev2_conv2d_transpose_13_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_16_kernel_m_read_readvariableop0savev2_adam_conv2d_16_bias_m_read_readvariableop2savev2_adam_conv2d_17_kernel_m_read_readvariableop0savev2_adam_conv2d_17_bias_m_read_readvariableop2savev2_adam_conv2d_18_kernel_m_read_readvariableop0savev2_adam_conv2d_18_bias_m_read_readvariableop2savev2_adam_conv2d_19_kernel_m_read_readvariableop0savev2_adam_conv2d_19_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_10_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_10_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_11_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_11_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_12_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_12_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_13_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_13_bias_m_read_readvariableop2savev2_adam_conv2d_16_kernel_v_read_readvariableop0savev2_adam_conv2d_16_bias_v_read_readvariableop2savev2_adam_conv2d_17_kernel_v_read_readvariableop0savev2_adam_conv2d_17_bias_v_read_readvariableop2savev2_adam_conv2d_18_kernel_v_read_readvariableop0savev2_adam_conv2d_18_bias_v_read_readvariableop2savev2_adam_conv2d_19_kernel_v_read_readvariableop0savev2_adam_conv2d_19_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_10_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_10_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_11_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_11_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_12_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_12_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_13_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	
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

identity_1Identity_1:output:0*
_input_shapesє
ё: : : : @:@:@::::::@:@: @: : :: : : : : : : : : : : : @:@:@::::::@:@: @: : :: : : @:@:@::::::@:@: @: : :: 2(
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
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::-)
'
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::
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
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::. *
(
_output_shapes
::!!

_output_shapes	
::."*
(
_output_shapes
::!#

_output_shapes	
::-$)
'
_output_shapes
:@: %

_output_shapes
:@:,&(
&
_output_shapes
: @: '

_output_shapes
: :,((
&
_output_shapes
: : )

_output_shapes
::,*(
&
_output_shapes
: : +

_output_shapes
: :,,(
&
_output_shapes
: @: -

_output_shapes
:@:-.)
'
_output_shapes
:@:!/

_output_shapes	
::.0*
(
_output_shapes
::!1

_output_shapes	
::.2*
(
_output_shapes
::!3

_output_shapes	
::-4)
'
_output_shapes
:@: 5

_output_shapes
:@:,6(
&
_output_shapes
: @: 7

_output_shapes
: :,8(
&
_output_shapes
: : 9

_output_shapes
:::

_output_shapes
: 
Ш
Ј
2__inference_conv2d_transpose_11_layer_call_fn_7614

inputs"
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallќ
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
GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_6242
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


л
&__inference_encoder_layer_call_fn_5984
input_9!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_5965x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџаа: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџаа
!
_user_specified_name	input_9
Ї
щ
A__inference_model_2_layer_call_and_return_conditional_losses_7054

inputsJ
0encoder_conv2d_16_conv2d_readvariableop_resource: ?
1encoder_conv2d_16_biasadd_readvariableop_resource: J
0encoder_conv2d_17_conv2d_readvariableop_resource: @?
1encoder_conv2d_17_biasadd_readvariableop_resource:@K
0encoder_conv2d_18_conv2d_readvariableop_resource:@@
1encoder_conv2d_18_biasadd_readvariableop_resource:	L
0encoder_conv2d_19_conv2d_readvariableop_resource:@
1encoder_conv2d_19_biasadd_readvariableop_resource:	`
Ddecoder_conv2d_transpose_10_conv2d_transpose_readvariableop_resource:J
;decoder_conv2d_transpose_10_biasadd_readvariableop_resource:	_
Ddecoder_conv2d_transpose_11_conv2d_transpose_readvariableop_resource:@I
;decoder_conv2d_transpose_11_biasadd_readvariableop_resource:@^
Ddecoder_conv2d_transpose_12_conv2d_transpose_readvariableop_resource: @I
;decoder_conv2d_transpose_12_biasadd_readvariableop_resource: ^
Ddecoder_conv2d_transpose_13_conv2d_transpose_readvariableop_resource: I
;decoder_conv2d_transpose_13_biasadd_readvariableop_resource:
identityЂ2decoder/conv2d_transpose_10/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_11/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_12/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOpЂ2decoder/conv2d_transpose_13/BiasAdd/ReadVariableOpЂ;decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOpЂ(encoder/conv2d_16/BiasAdd/ReadVariableOpЂ'encoder/conv2d_16/Conv2D/ReadVariableOpЂ(encoder/conv2d_17/BiasAdd/ReadVariableOpЂ'encoder/conv2d_17/Conv2D/ReadVariableOpЂ(encoder/conv2d_18/BiasAdd/ReadVariableOpЂ'encoder/conv2d_18/Conv2D/ReadVariableOpЂ(encoder/conv2d_19/BiasAdd/ReadVariableOpЂ'encoder/conv2d_19/Conv2D/ReadVariableOp 
'encoder/conv2d_16/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Н
encoder/conv2d_16/Conv2DConv2Dinputs/encoder/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh *
paddingSAME*
strides

(encoder/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Г
encoder/conv2d_16/BiasAddBiasAdd!encoder/conv2d_16/Conv2D:output:00encoder/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh |
encoder/conv2d_16/ReluRelu"encoder/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh  
'encoder/conv2d_17/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0л
encoder/conv2d_17/Conv2DConv2D$encoder/conv2d_16/Relu:activations:0/encoder/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@*
paddingSAME*
strides

(encoder/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
encoder/conv2d_17/BiasAddBiasAdd!encoder/conv2d_17/Conv2D:output:00encoder/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@|
encoder/conv2d_17/ReluRelu"encoder/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ44@Ё
'encoder/conv2d_18/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_18_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0м
encoder/conv2d_18/Conv2DConv2D$encoder/conv2d_17/Relu:activations:0/encoder/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

(encoder/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Д
encoder/conv2d_18/BiasAddBiasAdd!encoder/conv2d_18/Conv2D:output:00encoder/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ}
encoder/conv2d_18/ReluRelu"encoder/conv2d_18/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЂ
'encoder/conv2d_19/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0м
encoder/conv2d_19/Conv2DConv2D$encoder/conv2d_18/Relu:activations:0/encoder/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

(encoder/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Д
encoder/conv2d_19/BiasAddBiasAdd!encoder/conv2d_19/Conv2D:output:00encoder/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ}
encoder/conv2d_19/ReluRelu"encoder/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџu
!decoder/conv2d_transpose_10/ShapeShape$encoder/conv2d_19/Relu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_10/strided_sliceStridedSlice*decoder/conv2d_transpose_10/Shape:output:08decoder/conv2d_transpose_10/strided_slice/stack:output:0:decoder/conv2d_transpose_10/strided_slice/stack_1:output:0:decoder/conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#decoder/conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value	B :f
#decoder/conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value
B :
!decoder/conv2d_transpose_10/stackPack2decoder/conv2d_transpose_10/strided_slice:output:0,decoder/conv2d_transpose_10/stack/1:output:0,decoder/conv2d_transpose_10/stack/2:output:0,decoder/conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_10/strided_slice_1StridedSlice*decoder/conv2d_transpose_10/stack:output:0:decoder/conv2d_transpose_10/strided_slice_1/stack:output:0<decoder/conv2d_transpose_10/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЪ
;decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_10_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Н
,decoder/conv2d_transpose_10/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_10/stack:output:0Cdecoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0$encoder/conv2d_19/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
Ћ
2decoder/conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0м
#decoder/conv2d_transpose_10/BiasAddBiasAdd5decoder/conv2d_transpose_10/conv2d_transpose:output:0:decoder/conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ
 decoder/conv2d_transpose_10/ReluRelu,decoder/conv2d_transpose_10/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
!decoder/conv2d_transpose_11/ShapeShape.decoder/conv2d_transpose_10/Relu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_11/strided_sliceStridedSlice*decoder/conv2d_transpose_11/Shape:output:08decoder/conv2d_transpose_11/strided_slice/stack:output:0:decoder/conv2d_transpose_11/strided_slice/stack_1:output:0:decoder/conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value	B :4e
#decoder/conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value	B :4e
#decoder/conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@
!decoder/conv2d_transpose_11/stackPack2decoder/conv2d_transpose_11/strided_slice:output:0,decoder/conv2d_transpose_11/stack/1:output:0,decoder/conv2d_transpose_11/stack/2:output:0,decoder/conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_11/strided_slice_1StridedSlice*decoder/conv2d_transpose_11/stack:output:0:decoder/conv2d_transpose_11/strided_slice_1/stack:output:0<decoder/conv2d_transpose_11/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЩ
;decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_11_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0Ц
,decoder/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_11/stack:output:0Cdecoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_10/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ44@*
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0л
#decoder/conv2d_transpose_11/BiasAddBiasAdd5decoder/conv2d_transpose_11/conv2d_transpose:output:0:decoder/conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@
 decoder/conv2d_transpose_11/ReluRelu,decoder/conv2d_transpose_11/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ44@
!decoder/conv2d_transpose_12/ShapeShape.decoder/conv2d_transpose_11/Relu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_12/strided_sliceStridedSlice*decoder/conv2d_transpose_12/Shape:output:08decoder/conv2d_transpose_12/strided_slice/stack:output:0:decoder/conv2d_transpose_12/strided_slice/stack_1:output:0:decoder/conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#decoder/conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :he
#decoder/conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :he
#decoder/conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
!decoder/conv2d_transpose_12/stackPack2decoder/conv2d_transpose_12/strided_slice:output:0,decoder/conv2d_transpose_12/stack/1:output:0,decoder/conv2d_transpose_12/stack/2:output:0,decoder/conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_12/strided_slice_1StridedSlice*decoder/conv2d_transpose_12/stack:output:0:decoder/conv2d_transpose_12/strided_slice_1/stack:output:0<decoder/conv2d_transpose_12/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
;decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0Ц
,decoder/conv2d_transpose_12/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_12/stack:output:0Cdecoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_11/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџhh *
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0л
#decoder/conv2d_transpose_12/BiasAddBiasAdd5decoder/conv2d_transpose_12/conv2d_transpose:output:0:decoder/conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh 
 decoder/conv2d_transpose_12/ReluRelu,decoder/conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh 
!decoder/conv2d_transpose_13/ShapeShape.decoder/conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:y
/decoder/conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1decoder/conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1decoder/conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)decoder/conv2d_transpose_13/strided_sliceStridedSlice*decoder/conv2d_transpose_13/Shape:output:08decoder/conv2d_transpose_13/strided_slice/stack:output:0:decoder/conv2d_transpose_13/strided_slice/stack_1:output:0:decoder/conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#decoder/conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value
B :аf
#decoder/conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value
B :аe
#decoder/conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
!decoder/conv2d_transpose_13/stackPack2decoder/conv2d_transpose_13/strided_slice:output:0,decoder/conv2d_transpose_13/stack/1:output:0,decoder/conv2d_transpose_13/stack/2:output:0,decoder/conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:{
1decoder/conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3decoder/conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3decoder/conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
+decoder/conv2d_transpose_13/strided_slice_1StridedSlice*decoder/conv2d_transpose_13/stack:output:0:decoder/conv2d_transpose_13/strided_slice_1/stack:output:0<decoder/conv2d_transpose_13/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskШ
;decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ш
,decoder/conv2d_transpose_13/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_13/stack:output:0Cdecoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_12/Relu:activations:0*
T0*1
_output_shapes
:џџџџџџџџџаа*
paddingSAME*
strides
Њ
2decoder/conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0н
#decoder/conv2d_transpose_13/BiasAddBiasAdd5decoder/conv2d_transpose_13/conv2d_transpose:output:0:decoder/conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџаа
#decoder/conv2d_transpose_13/SigmoidSigmoid,decoder/conv2d_transpose_13/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџаа
IdentityIdentity'decoder/conv2d_transpose_13/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаац
NoOpNoOp3^decoder/conv2d_transpose_10/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_11/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_12/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_13/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOp)^encoder/conv2d_16/BiasAdd/ReadVariableOp(^encoder/conv2d_16/Conv2D/ReadVariableOp)^encoder/conv2d_17/BiasAdd/ReadVariableOp(^encoder/conv2d_17/Conv2D/ReadVariableOp)^encoder/conv2d_18/BiasAdd/ReadVariableOp(^encoder/conv2d_18/Conv2D/ReadVariableOp)^encoder/conv2d_19/BiasAdd/ReadVariableOp(^encoder/conv2d_19/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџаа: : : : : : : : : : : : : : : : 2h
2decoder/conv2d_transpose_10/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_10/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_10/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_11/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_11/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_11/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_12/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_12/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_12/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_13/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_13/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_13/conv2d_transpose/ReadVariableOp2T
(encoder/conv2d_16/BiasAdd/ReadVariableOp(encoder/conv2d_16/BiasAdd/ReadVariableOp2R
'encoder/conv2d_16/Conv2D/ReadVariableOp'encoder/conv2d_16/Conv2D/ReadVariableOp2T
(encoder/conv2d_17/BiasAdd/ReadVariableOp(encoder/conv2d_17/BiasAdd/ReadVariableOp2R
'encoder/conv2d_17/Conv2D/ReadVariableOp'encoder/conv2d_17/Conv2D/ReadVariableOp2T
(encoder/conv2d_18/BiasAdd/ReadVariableOp(encoder/conv2d_18/BiasAdd/ReadVariableOp2R
'encoder/conv2d_18/Conv2D/ReadVariableOp'encoder/conv2d_18/Conv2D/ReadVariableOp2T
(encoder/conv2d_19/BiasAdd/ReadVariableOp(encoder/conv2d_19/BiasAdd/ReadVariableOp2R
'encoder/conv2d_19/Conv2D/ReadVariableOp'encoder/conv2d_19/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs

ќ
C__inference_conv2d_17_layer_call_and_return_conditional_losses_5924

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ44@*
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
:џџџџџџџџџ44@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ44@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ44@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџhh : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџhh 
 
_user_specified_nameinputs


й
&__inference_decoder_layer_call_fn_7314

inputs#
unknown:
	unknown_0:	$
	unknown_1:@
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_6433y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


л
&__inference_decoder_layer_call_fn_6473
input_10#
unknown:
	unknown_0:	$
	unknown_1:@
	unknown_2:@#
	unknown_3: @
	unknown_4: #
	unknown_5: 
	unknown_6:
identityЂStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_6433y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_10
Љ
Ю
A__inference_model_2_layer_call_and_return_conditional_losses_6823
input_9&
encoder_6788: 
encoder_6790: &
encoder_6792: @
encoder_6794:@'
encoder_6796:@
encoder_6798:	(
encoder_6800:
encoder_6802:	(
decoder_6805:
decoder_6807:	'
decoder_6809:@
decoder_6811:@&
decoder_6813: @
decoder_6815: &
decoder_6817: 
decoder_6819:
identityЂdecoder/StatefulPartitionedCallЂencoder/StatefulPartitionedCallа
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_9encoder_6788encoder_6790encoder_6792encoder_6794encoder_6796encoder_6798encoder_6800encoder_6802*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_6071ђ
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_6805decoder_6807decoder_6809decoder_6811decoder_6813decoder_6815decoder_6817decoder_6819*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_6433
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџаа: : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџаа
!
_user_specified_name	input_9
І
Э
A__inference_model_2_layer_call_and_return_conditional_losses_6563

inputs&
encoder_6528: 
encoder_6530: &
encoder_6532: @
encoder_6534:@'
encoder_6536:@
encoder_6538:	(
encoder_6540:
encoder_6542:	(
decoder_6545:
decoder_6547:	'
decoder_6549:@
decoder_6551:@&
decoder_6553: @
decoder_6555: &
decoder_6557: 
decoder_6559:
identityЂdecoder/StatefulPartitionedCallЂencoder/StatefulPartitionedCallЯ
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_6528encoder_6530encoder_6532encoder_6534encoder_6536encoder_6538encoder_6540encoder_6542*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_5965ђ
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_6545decoder_6547decoder_6549decoder_6551decoder_6553decoder_6555decoder_6557decoder_6559*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџаа**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_decoder_layer_call_and_return_conditional_losses_6367
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџаа: : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs

ў
C__inference_conv2d_18_layer_call_and_return_conditional_losses_7542

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ44@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ44@
 
_user_specified_nameinputs


к
&__inference_encoder_layer_call_fn_7208

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	
identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_6071x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџаа: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs
Ю

A__inference_encoder_layer_call_and_return_conditional_losses_5965

inputs(
conv2d_16_5908: 
conv2d_16_5910: (
conv2d_17_5925: @
conv2d_17_5927:@)
conv2d_18_5942:@
conv2d_18_5944:	*
conv2d_19_5959:
conv2d_19_5961:	
identityЂ!conv2d_16/StatefulPartitionedCallЂ!conv2d_17/StatefulPartitionedCallЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallі
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_16_5908conv2d_16_5910*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_16_layer_call_and_return_conditional_losses_5907
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0conv2d_17_5925conv2d_17_5927*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ44@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_17_layer_call_and_return_conditional_losses_5924
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0conv2d_18_5942conv2d_18_5944*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_5941
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0conv2d_19_5959conv2d_19_5961*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_5958
IdentityIdentity*conv2d_19/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџж
NoOpNoOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџаа: : : : : : : : 2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs

л
&__inference_model_2_layer_call_fn_6942

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	$
	unknown_9:@

unknown_10:@$

unknown_11: @

unknown_12: $

unknown_13: 

unknown_14:
identityЂStatefulPartitionedCall
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
 *1
_output_shapes
:џџџџџџџџџаа*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_6675y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџаа: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs

л
&__inference_model_2_layer_call_fn_6905

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	$
	unknown_9:@

unknown_10:@$

unknown_11: @

unknown_12: $

unknown_13: 

unknown_14:
identityЂStatefulPartitionedCall
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
 *1
_output_shapes
:џџџџџџџџџаа*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_2_layer_call_and_return_conditional_losses_6563y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:џџџџџџџџџаа`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџаа: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџаа
 
_user_specified_nameinputs"Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*О
serving_defaultЊ
E
input_9:
serving_default_input_9:0џџџџџџџџџааE
decoder:
StatefulPartitionedCall:0џџџџџџџџџааtensorflow/serving/predict:ЧЛ
О
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
а
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
а
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_network

"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115"
trackable_list_wrapper

"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
Э
7trace_0
8trace_1
9trace_2
:trace_32т
&__inference_model_2_layer_call_fn_6598
&__inference_model_2_layer_call_fn_6905
&__inference_model_2_layer_call_fn_6942
&__inference_model_2_layer_call_fn_6747П
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
 z7trace_0z8trace_1z9trace_2z:trace_3
Й
;trace_0
<trace_1
=trace_2
>trace_32Ю
A__inference_model_2_layer_call_and_return_conditional_losses_7054
A__inference_model_2_layer_call_and_return_conditional_losses_7166
A__inference_model_2_layer_call_and_return_conditional_losses_6785
A__inference_model_2_layer_call_and_return_conditional_losses_6823П
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
 z;trace_0z<trace_1z=trace_2z>trace_3
ЪBЧ
__inference__wrapped_model_5889input_9"
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

?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rate"mк#mл$mм%mн&mо'mп(mр)mс*mт+mу,mф-mх.mц/mч0mш1mщ"vъ#vы$vь%vэ&vю'vя(v№)vё*vђ+vѓ,vє-vѕ.vі/vї0vј1vљ"
	optimizer
,
Dserving_default"
signature_map
н
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

"kernel
#bias
 K_jit_compiled_convolution_op"
_tf_keras_layer
н
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

$kernel
%bias
 R_jit_compiled_convolution_op"
_tf_keras_layer
н
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

&kernel
'bias
 Y_jit_compiled_convolution_op"
_tf_keras_layer
н
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

(kernel
)bias
 `_jit_compiled_convolution_op"
_tf_keras_layer
X
"0
#1
$2
%3
&4
'5
(6
)7"
trackable_list_wrapper
X
"0
#1
$2
%3
&4
'5
(6
)7"
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Э
ftrace_0
gtrace_1
htrace_2
itrace_32т
&__inference_encoder_layer_call_fn_5984
&__inference_encoder_layer_call_fn_7187
&__inference_encoder_layer_call_fn_7208
&__inference_encoder_layer_call_fn_6111П
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
 zftrace_0zgtrace_1zhtrace_2zitrace_3
Й
jtrace_0
ktrace_1
ltrace_2
mtrace_32Ю
A__inference_encoder_layer_call_and_return_conditional_losses_7240
A__inference_encoder_layer_call_and_return_conditional_losses_7272
A__inference_encoder_layer_call_and_return_conditional_losses_6135
A__inference_encoder_layer_call_and_return_conditional_losses_6159П
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
 zjtrace_0zktrace_1zltrace_2zmtrace_3
"
_tf_keras_input_layer
н
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

*kernel
+bias
 t_jit_compiled_convolution_op"
_tf_keras_layer
н
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

,kernel
-bias
 {_jit_compiled_convolution_op"
_tf_keras_layer
р
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

.kernel
/bias
!_jit_compiled_convolution_op"
_tf_keras_layer
ф
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

0kernel
1bias
!_jit_compiled_convolution_op"
_tf_keras_layer
X
*0
+1
,2
-3
.4
/5
06
17"
trackable_list_wrapper
X
*0
+1
,2
-3
.4
/5
06
17"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
е
trace_0
trace_1
trace_2
trace_32т
&__inference_decoder_layer_call_fn_6386
&__inference_decoder_layer_call_fn_7293
&__inference_decoder_layer_call_fn_7314
&__inference_decoder_layer_call_fn_6473П
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
 ztrace_0ztrace_1ztrace_2ztrace_3
С
trace_0
trace_1
trace_2
trace_32Ю
A__inference_decoder_layer_call_and_return_conditional_losses_7398
A__inference_decoder_layer_call_and_return_conditional_losses_7482
A__inference_decoder_layer_call_and_return_conditional_losses_6497
A__inference_decoder_layer_call_and_return_conditional_losses_6521П
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
 ztrace_0ztrace_1ztrace_2ztrace_3
*:( 2conv2d_16/kernel
: 2conv2d_16/bias
*:( @2conv2d_17/kernel
:@2conv2d_17/bias
+:)@2conv2d_18/kernel
:2conv2d_18/bias
,:*2conv2d_19/kernel
:2conv2d_19/bias
6:42conv2d_transpose_10/kernel
':%2conv2d_transpose_10/bias
5:3@2conv2d_transpose_11/kernel
&:$@2conv2d_transpose_11/bias
4:2 @2conv2d_transpose_12/kernel
&:$ 2conv2d_transpose_12/bias
4:2 2conv2d_transpose_13/kernel
&:$2conv2d_transpose_13/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
јBѕ
&__inference_model_2_layer_call_fn_6598input_9"П
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
їBє
&__inference_model_2_layer_call_fn_6905inputs"П
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
їBє
&__inference_model_2_layer_call_fn_6942inputs"П
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
јBѕ
&__inference_model_2_layer_call_fn_6747input_9"П
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
B
A__inference_model_2_layer_call_and_return_conditional_losses_7054inputs"П
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
B
A__inference_model_2_layer_call_and_return_conditional_losses_7166inputs"П
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
B
A__inference_model_2_layer_call_and_return_conditional_losses_6785input_9"П
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
B
A__inference_model_2_layer_call_and_return_conditional_losses_6823input_9"П
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ЩBЦ
"__inference_signature_wrapper_6868input_9"
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
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
(__inference_conv2d_16_layer_call_fn_7491Ђ
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
 ztrace_0

trace_02ъ
C__inference_conv2d_16_layer_call_and_return_conditional_losses_7502Ђ
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
 ztrace_0
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
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
ю
Ѕtrace_02Я
(__inference_conv2d_17_layer_call_fn_7511Ђ
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
 zЅtrace_0

Іtrace_02ъ
C__inference_conv2d_17_layer_call_and_return_conditional_losses_7522Ђ
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
 zІtrace_0
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
В
Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
ю
Ќtrace_02Я
(__inference_conv2d_18_layer_call_fn_7531Ђ
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
 zЌtrace_0

­trace_02ъ
C__inference_conv2d_18_layer_call_and_return_conditional_losses_7542Ђ
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
 z­trace_0
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
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
ю
Гtrace_02Я
(__inference_conv2d_19_layer_call_fn_7551Ђ
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
 zГtrace_0

Дtrace_02ъ
C__inference_conv2d_19_layer_call_and_return_conditional_losses_7562Ђ
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
 zДtrace_0
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
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
јBѕ
&__inference_encoder_layer_call_fn_5984input_9"П
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
їBє
&__inference_encoder_layer_call_fn_7187inputs"П
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
їBє
&__inference_encoder_layer_call_fn_7208inputs"П
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
јBѕ
&__inference_encoder_layer_call_fn_6111input_9"П
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
B
A__inference_encoder_layer_call_and_return_conditional_losses_7240inputs"П
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
B
A__inference_encoder_layer_call_and_return_conditional_losses_7272inputs"П
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
B
A__inference_encoder_layer_call_and_return_conditional_losses_6135input_9"П
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
B
A__inference_encoder_layer_call_and_return_conditional_losses_6159input_9"П
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
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
ј
Кtrace_02й
2__inference_conv2d_transpose_10_layer_call_fn_7571Ђ
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
 zКtrace_0

Лtrace_02є
M__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_7605Ђ
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
 zЛtrace_0
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
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
ј
Сtrace_02й
2__inference_conv2d_transpose_11_layer_call_fn_7614Ђ
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
 zСtrace_0

Тtrace_02є
M__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_7648Ђ
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
 zТtrace_0
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
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ј
Шtrace_02й
2__inference_conv2d_transpose_12_layer_call_fn_7657Ђ
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
 zШtrace_0

Щtrace_02є
M__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_7691Ђ
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
 zЩtrace_0
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
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ъnon_trainable_variables
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ј
Яtrace_02й
2__inference_conv2d_transpose_13_layer_call_fn_7700Ђ
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
 zЯtrace_0

аtrace_02є
M__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_7734Ђ
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
 zаtrace_0
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
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
љBі
&__inference_decoder_layer_call_fn_6386input_10"П
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
їBє
&__inference_decoder_layer_call_fn_7293inputs"П
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
їBє
&__inference_decoder_layer_call_fn_7314inputs"П
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
&__inference_decoder_layer_call_fn_6473input_10"П
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
B
A__inference_decoder_layer_call_and_return_conditional_losses_7398inputs"П
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
B
A__inference_decoder_layer_call_and_return_conditional_losses_7482inputs"П
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
A__inference_decoder_layer_call_and_return_conditional_losses_6497input_10"П
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
A__inference_decoder_layer_call_and_return_conditional_losses_6521input_10"П
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
R
б	variables
в	keras_api

гtotal

дcount"
_tf_keras_metric
c
е	variables
ж	keras_api

зtotal

иcount
й
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
мBй
(__inference_conv2d_16_layer_call_fn_7491inputs"Ђ
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
їBє
C__inference_conv2d_16_layer_call_and_return_conditional_losses_7502inputs"Ђ
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
мBй
(__inference_conv2d_17_layer_call_fn_7511inputs"Ђ
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
їBє
C__inference_conv2d_17_layer_call_and_return_conditional_losses_7522inputs"Ђ
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
мBй
(__inference_conv2d_18_layer_call_fn_7531inputs"Ђ
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
їBє
C__inference_conv2d_18_layer_call_and_return_conditional_losses_7542inputs"Ђ
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
мBй
(__inference_conv2d_19_layer_call_fn_7551inputs"Ђ
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
їBє
C__inference_conv2d_19_layer_call_and_return_conditional_losses_7562inputs"Ђ
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
цBу
2__inference_conv2d_transpose_10_layer_call_fn_7571inputs"Ђ
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
Bў
M__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_7605inputs"Ђ
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
цBу
2__inference_conv2d_transpose_11_layer_call_fn_7614inputs"Ђ
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
Bў
M__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_7648inputs"Ђ
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
цBу
2__inference_conv2d_transpose_12_layer_call_fn_7657inputs"Ђ
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
Bў
M__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_7691inputs"Ђ
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
цBу
2__inference_conv2d_transpose_13_layer_call_fn_7700inputs"Ђ
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
Bў
M__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_7734inputs"Ђ
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
г0
д1"
trackable_list_wrapper
.
б	variables"
_generic_user_object
:  (2total
:  (2count
0
з0
и1"
trackable_list_wrapper
.
е	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:- 2Adam/conv2d_16/kernel/m
!: 2Adam/conv2d_16/bias/m
/:- @2Adam/conv2d_17/kernel/m
!:@2Adam/conv2d_17/bias/m
0:.@2Adam/conv2d_18/kernel/m
": 2Adam/conv2d_18/bias/m
1:/2Adam/conv2d_19/kernel/m
": 2Adam/conv2d_19/bias/m
;:92!Adam/conv2d_transpose_10/kernel/m
,:*2Adam/conv2d_transpose_10/bias/m
::8@2!Adam/conv2d_transpose_11/kernel/m
+:)@2Adam/conv2d_transpose_11/bias/m
9:7 @2!Adam/conv2d_transpose_12/kernel/m
+:) 2Adam/conv2d_transpose_12/bias/m
9:7 2!Adam/conv2d_transpose_13/kernel/m
+:)2Adam/conv2d_transpose_13/bias/m
/:- 2Adam/conv2d_16/kernel/v
!: 2Adam/conv2d_16/bias/v
/:- @2Adam/conv2d_17/kernel/v
!:@2Adam/conv2d_17/bias/v
0:.@2Adam/conv2d_18/kernel/v
": 2Adam/conv2d_18/bias/v
1:/2Adam/conv2d_19/kernel/v
": 2Adam/conv2d_19/bias/v
;:92!Adam/conv2d_transpose_10/kernel/v
,:*2Adam/conv2d_transpose_10/bias/v
::8@2!Adam/conv2d_transpose_11/kernel/v
+:)@2Adam/conv2d_transpose_11/bias/v
9:7 @2!Adam/conv2d_transpose_12/kernel/v
+:) 2Adam/conv2d_transpose_12/bias/v
9:7 2!Adam/conv2d_transpose_13/kernel/v
+:)2Adam/conv2d_transpose_13/bias/vЏ
__inference__wrapped_model_5889"#$%&'()*+,-./01:Ђ7
0Ђ-
+(
input_9џџџџџџџџџаа
Њ ";Њ8
6
decoder+(
decoderџџџџџџџџџааЕ
C__inference_conv2d_16_layer_call_and_return_conditional_losses_7502n"#9Ђ6
/Ђ,
*'
inputsџџџџџџџџџаа
Њ "-Ђ*
# 
0џџџџџџџџџhh 
 
(__inference_conv2d_16_layer_call_fn_7491a"#9Ђ6
/Ђ,
*'
inputsџџџџџџџџџаа
Њ " џџџџџџџџџhh Г
C__inference_conv2d_17_layer_call_and_return_conditional_losses_7522l$%7Ђ4
-Ђ*
(%
inputsџџџџџџџџџhh 
Њ "-Ђ*
# 
0џџџџџџџџџ44@
 
(__inference_conv2d_17_layer_call_fn_7511_$%7Ђ4
-Ђ*
(%
inputsџџџџџџџџџhh 
Њ " џџџџџџџџџ44@Д
C__inference_conv2d_18_layer_call_and_return_conditional_losses_7542m&'7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ44@
Њ ".Ђ+
$!
0џџџџџџџџџ
 
(__inference_conv2d_18_layer_call_fn_7531`&'7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ44@
Њ "!џџџџџџџџџЕ
C__inference_conv2d_19_layer_call_and_return_conditional_losses_7562n()8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
(__inference_conv2d_19_layer_call_fn_7551a()8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџф
M__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_7605*+JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "@Ђ=
63
0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 М
2__inference_conv2d_transpose_10_layer_call_fn_7571*+JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "30,џџџџџџџџџџџџџџџџџџџџџџџџџџџу
M__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_7648,-JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Л
2__inference_conv2d_transpose_11_layer_call_fn_7614,-JЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@т
M__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_7691./IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 К
2__inference_conv2d_transpose_12_layer_call_fn_7657./IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ т
M__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_773401IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 К
2__inference_conv2d_transpose_13_layer_call_fn_770001IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџФ
A__inference_decoder_layer_call_and_return_conditional_losses_6497*+,-./01BЂ?
8Ђ5
+(
input_10џџџџџџџџџ
p 

 
Њ "/Ђ,
%"
0џџџџџџџџџаа
 Ф
A__inference_decoder_layer_call_and_return_conditional_losses_6521*+,-./01BЂ?
8Ђ5
+(
input_10џџџџџџџџџ
p

 
Њ "/Ђ,
%"
0џџџџџџџџџаа
 Т
A__inference_decoder_layer_call_and_return_conditional_losses_7398}*+,-./01@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ
p 

 
Њ "/Ђ,
%"
0џџџџџџџџџаа
 Т
A__inference_decoder_layer_call_and_return_conditional_losses_7482}*+,-./01@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ
p

 
Њ "/Ђ,
%"
0џџџџџџџџџаа
 
&__inference_decoder_layer_call_fn_6386r*+,-./01BЂ?
8Ђ5
+(
input_10џџџџџџџџџ
p 

 
Њ ""џџџџџџџџџаа
&__inference_decoder_layer_call_fn_6473r*+,-./01BЂ?
8Ђ5
+(
input_10џџџџџџџџџ
p

 
Њ ""џџџџџџџџџаа
&__inference_decoder_layer_call_fn_7293p*+,-./01@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ
p 

 
Њ ""џџџџџџџџџаа
&__inference_decoder_layer_call_fn_7314p*+,-./01@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ
p

 
Њ ""џџџџџџџџџааУ
A__inference_encoder_layer_call_and_return_conditional_losses_6135~"#$%&'()BЂ?
8Ђ5
+(
input_9џџџџџџџџџаа
p 

 
Њ ".Ђ+
$!
0џџџџџџџџџ
 У
A__inference_encoder_layer_call_and_return_conditional_losses_6159~"#$%&'()BЂ?
8Ђ5
+(
input_9џџџџџџџџџаа
p

 
Њ ".Ђ+
$!
0џџџџџџџџџ
 Т
A__inference_encoder_layer_call_and_return_conditional_losses_7240}"#$%&'()AЂ>
7Ђ4
*'
inputsџџџџџџџџџаа
p 

 
Њ ".Ђ+
$!
0џџџџџџџџџ
 Т
A__inference_encoder_layer_call_and_return_conditional_losses_7272}"#$%&'()AЂ>
7Ђ4
*'
inputsџџџџџџџџџаа
p

 
Њ ".Ђ+
$!
0џџџџџџџџџ
 
&__inference_encoder_layer_call_fn_5984q"#$%&'()BЂ?
8Ђ5
+(
input_9џџџџџџџџџаа
p 

 
Њ "!џџџџџџџџџ
&__inference_encoder_layer_call_fn_6111q"#$%&'()BЂ?
8Ђ5
+(
input_9џџџџџџџџџаа
p

 
Њ "!џџџџџџџџџ
&__inference_encoder_layer_call_fn_7187p"#$%&'()AЂ>
7Ђ4
*'
inputsџџџџџџџџџаа
p 

 
Њ "!џџџџџџџџџ
&__inference_encoder_layer_call_fn_7208p"#$%&'()AЂ>
7Ђ4
*'
inputsџџџџџџџџџаа
p

 
Њ "!џџџџџџџџџЭ
A__inference_model_2_layer_call_and_return_conditional_losses_6785"#$%&'()*+,-./01BЂ?
8Ђ5
+(
input_9џџџџџџџџџаа
p 

 
Њ "/Ђ,
%"
0џџџџџџџџџаа
 Э
A__inference_model_2_layer_call_and_return_conditional_losses_6823"#$%&'()*+,-./01BЂ?
8Ђ5
+(
input_9џџџџџџџџџаа
p

 
Њ "/Ђ,
%"
0џџџџџџџџџаа
 Ь
A__inference_model_2_layer_call_and_return_conditional_losses_7054"#$%&'()*+,-./01AЂ>
7Ђ4
*'
inputsџџџџџџџџџаа
p 

 
Њ "/Ђ,
%"
0џџџџџџџџџаа
 Ь
A__inference_model_2_layer_call_and_return_conditional_losses_7166"#$%&'()*+,-./01AЂ>
7Ђ4
*'
inputsџџџџџџџџџаа
p

 
Њ "/Ђ,
%"
0џџџџџџџџџаа
 Є
&__inference_model_2_layer_call_fn_6598z"#$%&'()*+,-./01BЂ?
8Ђ5
+(
input_9џџџџџџџџџаа
p 

 
Њ ""џџџџџџџџџааЄ
&__inference_model_2_layer_call_fn_6747z"#$%&'()*+,-./01BЂ?
8Ђ5
+(
input_9џџџџџџџџџаа
p

 
Њ ""џџџџџџџџџааЃ
&__inference_model_2_layer_call_fn_6905y"#$%&'()*+,-./01AЂ>
7Ђ4
*'
inputsџџџџџџџџџаа
p 

 
Њ ""џџџџџџџџџааЃ
&__inference_model_2_layer_call_fn_6942y"#$%&'()*+,-./01AЂ>
7Ђ4
*'
inputsџџџџџџџџџаа
p

 
Њ ""џџџџџџџџџааН
"__inference_signature_wrapper_6868"#$%&'()*+,-./01EЂB
Ђ 
;Њ8
6
input_9+(
input_9џџџџџџџџџаа";Њ8
6
decoder+(
decoderџџџџџџџџџаа