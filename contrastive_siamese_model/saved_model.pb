??$
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
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
?
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
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
?
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
?
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
executor_typestring ?
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??!
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
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:@*
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
?
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_6/gamma
?
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_6/beta
?
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_6/moving_mean
?
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_6/moving_variance
?
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?* 
shared_nameconv2d_7/kernel
|
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*'
_output_shapes
:@?*
dtype0
s
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_7/bias
l
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_7/gamma
?
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_7/beta
?
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes	
:?*
dtype0
?
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_7/moving_mean
?
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_7/moving_variance
?
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_8/kernel
}
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*(
_output_shapes
:??*
dtype0
s
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_8/bias
l
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_8/gamma
?
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_8/beta
?
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes	
:?*
dtype0
?
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_8/moving_mean
?
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_8/moving_variance
?
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_9/kernel
}
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*(
_output_shapes
:??*
dtype0
s
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_9/bias
l
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_9/gamma
?
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_9/beta
?
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes	
:?*
dtype0
?
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_9/moving_mean
?
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_9/moving_variance
?
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
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
?
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_6/kernel/m
?
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
:@*
dtype0
?
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_6/gamma/m
?
6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/m*
_output_shapes
:@*
dtype0
?
!Adam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_6/beta/m
?
5Adam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdam/conv2d_7/kernel/m
?
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_7/bias/m
z
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_7/gamma/m
?
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_7/beta/m
?
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameAdam/conv2d_8/kernel/m
?
*Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_8/bias/m
z
(Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_8/gamma/m
?
6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/m*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_8/beta/m
?
5Adam/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameAdam/conv2d_9/kernel/m
?
*Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_9/bias/m
z
(Adam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_9/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_9/gamma/m
?
6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/m*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_9/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_9/beta/m
?
5Adam/batch_normalization_9/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_6/kernel/v
?
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
:@*
dtype0
?
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_6/gamma/v
?
6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/v*
_output_shapes
:@*
dtype0
?
!Adam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_6/beta/v
?
5Adam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdam/conv2d_7/kernel/v
?
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_7/bias/v
z
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_7/gamma/v
?
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_7/beta/v
?
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameAdam/conv2d_8/kernel/v
?
*Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_8/bias/v
z
(Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_8/gamma/v
?
6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_8/gamma/v*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_8/beta/v
?
5Adam/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_8/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameAdam/conv2d_9/kernel/v
?
*Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_9/bias/v
z
(Adam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_9/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_9/gamma/v
?
6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_9/gamma/v*
_output_shapes	
:?*
dtype0
?
!Adam/batch_normalization_9/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/batch_normalization_9/beta/v
?
5Adam/batch_normalization_9/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_9/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?|
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?{
value?{B?{ B?{
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
 
 
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
layer-8
layer_with_weights-4
layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
	variables
regularization_losses
 trainable_variables
!	keras_api
R
"	variables
#regularization_losses
$trainable_variables
%	keras_api
?
&iter

'beta_1

(beta_2
	)decay
*learning_rate+m?,m?-m?.m?1m?2m?3m?4m?7m?8m?9m?:m?=m?>m??m?@m?Cm?Dm?+v?,v?-v?.v?1v?2v?3v?4v?7v?8v?9v?:v?=v?>v??v?@v?Cv?Dv?
?
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
 
?
+0
,1
-2
.3
14
25
36
47
78
89
910
:11
=12
>13
?14
@15
C16
D17
?
Elayer_metrics

Flayers
Glayer_regularization_losses
	variables
Hnon_trainable_variables
Imetrics
regularization_losses
trainable_variables
 
 
h

+kernel
,bias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
R
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
?
Raxis
	-gamma
.beta
/moving_mean
0moving_variance
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
R
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
h

1kernel
2bias
[	variables
\regularization_losses
]trainable_variables
^	keras_api
R
_	variables
`regularization_losses
atrainable_variables
b	keras_api
?
caxis
	3gamma
4beta
5moving_mean
6moving_variance
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
R
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
h

7kernel
8bias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
R
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
?
taxis
	9gamma
:beta
;moving_mean
<moving_variance
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
R
y	variables
zregularization_losses
{trainable_variables
|	keras_api
i

=kernel
>bias
}	variables
~regularization_losses
trainable_variables
?	keras_api
?
	?axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Ckernel
Dbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
 
?
+0
,1
-2
.3
14
25
36
47
78
89
910
:11
=12
>13
?14
@15
C16
D17
?
?layer_metrics
?layers
 ?layer_regularization_losses
	variables
?non_trainable_variables
?metrics
regularization_losses
 trainable_variables
 
 
 
?
?layer_metrics
?layers
 ?layer_regularization_losses
"	variables
?non_trainable_variables
?metrics
#regularization_losses
$trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_6/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_6/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_6/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_6/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_6/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_6/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_7/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_7/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_7/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_7/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_7/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_7/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_8/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_8/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_8/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_8/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_8/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_8/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_9/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_9/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_9/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_9/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_9/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_9/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_1/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_1/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3
 
8
/0
01
52
63
;4
<5
A6
B7

?0

+0
,1
 

+0
,1
?
?layer_metrics
?layers
 ?layer_regularization_losses
J	variables
?non_trainable_variables
?metrics
Kregularization_losses
Ltrainable_variables
 
 
 
?
?layer_metrics
?layers
 ?layer_regularization_losses
N	variables
?non_trainable_variables
?metrics
Oregularization_losses
Ptrainable_variables
 

-0
.1
/2
03
 

-0
.1
?
?layer_metrics
?layers
 ?layer_regularization_losses
S	variables
?non_trainable_variables
?metrics
Tregularization_losses
Utrainable_variables
 
 
 
?
?layer_metrics
?layers
 ?layer_regularization_losses
W	variables
?non_trainable_variables
?metrics
Xregularization_losses
Ytrainable_variables

10
21
 

10
21
?
?layer_metrics
?layers
 ?layer_regularization_losses
[	variables
?non_trainable_variables
?metrics
\regularization_losses
]trainable_variables
 
 
 
?
?layer_metrics
?layers
 ?layer_regularization_losses
_	variables
?non_trainable_variables
?metrics
`regularization_losses
atrainable_variables
 

30
41
52
63
 

30
41
?
?layer_metrics
?layers
 ?layer_regularization_losses
d	variables
?non_trainable_variables
?metrics
eregularization_losses
ftrainable_variables
 
 
 
?
?layer_metrics
?layers
 ?layer_regularization_losses
h	variables
?non_trainable_variables
?metrics
iregularization_losses
jtrainable_variables

70
81
 

70
81
?
?layer_metrics
?layers
 ?layer_regularization_losses
l	variables
?non_trainable_variables
?metrics
mregularization_losses
ntrainable_variables
 
 
 
?
?layer_metrics
?layers
 ?layer_regularization_losses
p	variables
?non_trainable_variables
?metrics
qregularization_losses
rtrainable_variables
 

90
:1
;2
<3
 

90
:1
?
?layer_metrics
?layers
 ?layer_regularization_losses
u	variables
?non_trainable_variables
?metrics
vregularization_losses
wtrainable_variables
 
 
 
?
?layer_metrics
?layers
 ?layer_regularization_losses
y	variables
?non_trainable_variables
?metrics
zregularization_losses
{trainable_variables

=0
>1
 

=0
>1
?
?layer_metrics
?layers
 ?layer_regularization_losses
}	variables
?non_trainable_variables
?metrics
~regularization_losses
trainable_variables
 

?0
@1
A2
B3
 

?0
@1
?
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
?regularization_losses
?trainable_variables
 
 
 
?
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
?regularization_losses
?trainable_variables
 
 
 
?
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
?regularization_losses
?trainable_variables
 
 
 
?
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
?regularization_losses
?trainable_variables

C0
D1
 

C0
D1
?
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
?regularization_losses
?trainable_variables
 
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 
8
/0
01
52
63
;4
<5
A6
B7
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 

/0
01
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

50
61
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

;0
<1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
nl
VARIABLE_VALUEAdam/conv2d_6/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_6/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_6/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_7/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_7/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_8/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_8/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_8/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_8/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_9/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_9/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_9/gamma/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_9/beta/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_1/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_1/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_6/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_6/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_6/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_7/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_7/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_8/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_8/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_8/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_8/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_9/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_9/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_9/gamma/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_9/beta/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_1/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_1/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_4Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
serving_default_input_5Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4serving_default_input_5conv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_9/kernelconv2d_9/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variancedense_1/kerneldense_1/bias*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *,
f'R%
#__inference_signature_wrapper_68186
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_6/beta/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp*Adam/conv2d_8/kernel/m/Read/ReadVariableOp(Adam/conv2d_8/bias/m/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_8/beta/m/Read/ReadVariableOp*Adam/conv2d_9/kernel/m/Read/ReadVariableOp(Adam/conv2d_9/bias/m/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_9/beta/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_6/beta/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp*Adam/conv2d_8/kernel/v/Read/ReadVariableOp(Adam/conv2d_8/bias/v/Read/ReadVariableOp6Adam/batch_normalization_8/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_8/beta/v/Read/ReadVariableOp*Adam/conv2d_9/kernel/v/Read/ReadVariableOp(Adam/conv2d_9/bias/v/Read/ReadVariableOp6Adam/batch_normalization_9/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_9/beta/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*R
TinK
I2G	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *'
f"R 
__inference__traced_save_70160
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_9/kernelconv2d_9/biasbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variancedense_1/kerneldense_1/biastotalcountAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/m"Adam/batch_normalization_6/gamma/m!Adam/batch_normalization_6/beta/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/conv2d_8/kernel/mAdam/conv2d_8/bias/m"Adam/batch_normalization_8/gamma/m!Adam/batch_normalization_8/beta/mAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/m"Adam/batch_normalization_9/gamma/m!Adam/batch_normalization_9/beta/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/v"Adam/batch_normalization_6/gamma/v!Adam/batch_normalization_6/beta/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/v"Adam/batch_normalization_8/gamma/v!Adam/batch_normalization_8/beta/vAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/v"Adam/batch_normalization_9/gamma/v!Adam/batch_normalization_9/beta/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*Q
TinJ
H2F*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? **
f%R#
!__inference__traced_restore_70377??
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69409

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_68838

inputsA
'conv2d_6_conv2d_readvariableop_resource:@6
(conv2d_6_biasadd_readvariableop_resource:@;
-batch_normalization_6_readvariableop_resource:@=
/batch_normalization_6_readvariableop_1_resource:@L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_7_conv2d_readvariableop_resource:@?7
(conv2d_7_biasadd_readvariableop_resource:	?<
-batch_normalization_7_readvariableop_resource:	?>
/batch_normalization_7_readvariableop_1_resource:	?M
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	?C
'conv2d_8_conv2d_readvariableop_resource:??7
(conv2d_8_biasadd_readvariableop_resource:	?<
-batch_normalization_8_readvariableop_resource:	?>
/batch_normalization_8_readvariableop_1_resource:	?M
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	?C
'conv2d_9_conv2d_readvariableop_resource:??7
(conv2d_9_biasadd_readvariableop_resource:	?<
-batch_normalization_9_readvariableop_resource:	?>
/batch_normalization_9_readvariableop_1_resource:	?M
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?
identity??5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_9/ReadVariableOp?&batch_normalization_9/ReadVariableOp_1?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv2d_6/BiasAdd}
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
conv2d_6/Relu?
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_6/MaxPool:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3?
dropout_6/IdentityIdentity*batch_normalization_6/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
dropout_6/Identity?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Ddropout_6/Identity:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
conv2d_7/BiasAdd~
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
conv2d_7/Relu?
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu:activations:0*0
_output_shapes
:?????????@@?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPool?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_7/MaxPool:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3?
dropout_7/IdentityIdentity*batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@@?2
dropout_7/Identity?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Ddropout_7/Identity:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2
conv2d_8/BiasAdd|
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
conv2d_8/Relu?
max_pooling2d_8/MaxPoolMaxPoolconv2d_8/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_8/MaxPool?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_8/MaxPool:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3?
dropout_8/IdentityIdentity*batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
dropout_8/Identity?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2DConv2Ddropout_8/Identity:output:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv2d_9/Conv2D?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2
conv2d_9/BiasAdd|
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?2
conv2d_9/Relu?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_9/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 2(
&batch_normalization_9/FusedBatchNormV3?
max_pooling2d_9/MaxPoolMaxPool*batch_normalization_9/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPool?
dropout_9/IdentityIdentity max_pooling2d_9/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
dropout_9/Identity?
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_1/Mean/reduction_indices?
global_average_pooling2d_1/MeanMeandropout_9/Identity:output:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2!
global_average_pooling2d_1/Mean?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul(global_average_pooling2d_1/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddt
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?	
NoOpNoOp6^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_66512

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????@2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
'__inference_model_2_layer_call_fn_69027

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:
??

unknown_24:	?
identity??StatefulPartitionedCall?
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
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_667092
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
V
:__inference_global_average_pooling2d_1_layer_call_fn_69910

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *^
fYRW
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_666902
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_69531

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_7_layer_call_fn_69440

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_660262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?+
B__inference_model_3_layer_call_and_return_conditional_losses_68618
inputs_0
inputs_1I
/model_2_conv2d_6_conv2d_readvariableop_resource:@>
0model_2_conv2d_6_biasadd_readvariableop_resource:@C
5model_2_batch_normalization_6_readvariableop_resource:@E
7model_2_batch_normalization_6_readvariableop_1_resource:@T
Fmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@V
Hmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@J
/model_2_conv2d_7_conv2d_readvariableop_resource:@??
0model_2_conv2d_7_biasadd_readvariableop_resource:	?D
5model_2_batch_normalization_7_readvariableop_resource:	?F
7model_2_batch_normalization_7_readvariableop_1_resource:	?U
Fmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	?W
Hmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	?K
/model_2_conv2d_8_conv2d_readvariableop_resource:???
0model_2_conv2d_8_biasadd_readvariableop_resource:	?D
5model_2_batch_normalization_8_readvariableop_resource:	?F
7model_2_batch_normalization_8_readvariableop_1_resource:	?U
Fmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	?W
Hmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	?K
/model_2_conv2d_9_conv2d_readvariableop_resource:???
0model_2_conv2d_9_biasadd_readvariableop_resource:	?D
5model_2_batch_normalization_9_readvariableop_resource:	?F
7model_2_batch_normalization_9_readvariableop_1_resource:	?U
Fmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	?W
Hmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	?B
.model_2_dense_1_matmul_readvariableop_resource:
??>
/model_2_dense_1_biasadd_readvariableop_resource:	?
identity??,model_2/batch_normalization_6/AssignNewValue?.model_2/batch_normalization_6/AssignNewValue_1?.model_2/batch_normalization_6/AssignNewValue_2?.model_2/batch_normalization_6/AssignNewValue_3?=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp??model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1??model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp?Amodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1?,model_2/batch_normalization_6/ReadVariableOp?.model_2/batch_normalization_6/ReadVariableOp_1?.model_2/batch_normalization_6/ReadVariableOp_2?.model_2/batch_normalization_6/ReadVariableOp_3?,model_2/batch_normalization_7/AssignNewValue?.model_2/batch_normalization_7/AssignNewValue_1?.model_2/batch_normalization_7/AssignNewValue_2?.model_2/batch_normalization_7/AssignNewValue_3?=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp??model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1??model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp?Amodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1?,model_2/batch_normalization_7/ReadVariableOp?.model_2/batch_normalization_7/ReadVariableOp_1?.model_2/batch_normalization_7/ReadVariableOp_2?.model_2/batch_normalization_7/ReadVariableOp_3?,model_2/batch_normalization_8/AssignNewValue?.model_2/batch_normalization_8/AssignNewValue_1?.model_2/batch_normalization_8/AssignNewValue_2?.model_2/batch_normalization_8/AssignNewValue_3?=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp??model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1??model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp?Amodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1?,model_2/batch_normalization_8/ReadVariableOp?.model_2/batch_normalization_8/ReadVariableOp_1?.model_2/batch_normalization_8/ReadVariableOp_2?.model_2/batch_normalization_8/ReadVariableOp_3?,model_2/batch_normalization_9/AssignNewValue?.model_2/batch_normalization_9/AssignNewValue_1?.model_2/batch_normalization_9/AssignNewValue_2?.model_2/batch_normalization_9/AssignNewValue_3?=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp??model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1??model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp?Amodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1?,model_2/batch_normalization_9/ReadVariableOp?.model_2/batch_normalization_9/ReadVariableOp_1?.model_2/batch_normalization_9/ReadVariableOp_2?.model_2/batch_normalization_9/ReadVariableOp_3?'model_2/conv2d_6/BiasAdd/ReadVariableOp?)model_2/conv2d_6/BiasAdd_1/ReadVariableOp?&model_2/conv2d_6/Conv2D/ReadVariableOp?(model_2/conv2d_6/Conv2D_1/ReadVariableOp?'model_2/conv2d_7/BiasAdd/ReadVariableOp?)model_2/conv2d_7/BiasAdd_1/ReadVariableOp?&model_2/conv2d_7/Conv2D/ReadVariableOp?(model_2/conv2d_7/Conv2D_1/ReadVariableOp?'model_2/conv2d_8/BiasAdd/ReadVariableOp?)model_2/conv2d_8/BiasAdd_1/ReadVariableOp?&model_2/conv2d_8/Conv2D/ReadVariableOp?(model_2/conv2d_8/Conv2D_1/ReadVariableOp?'model_2/conv2d_9/BiasAdd/ReadVariableOp?)model_2/conv2d_9/BiasAdd_1/ReadVariableOp?&model_2/conv2d_9/Conv2D/ReadVariableOp?(model_2/conv2d_9/Conv2D_1/ReadVariableOp?&model_2/dense_1/BiasAdd/ReadVariableOp?(model_2/dense_1/BiasAdd_1/ReadVariableOp?%model_2/dense_1/MatMul/ReadVariableOp?'model_2/dense_1/MatMul_1/ReadVariableOp?
&model_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&model_2/conv2d_6/Conv2D/ReadVariableOp?
model_2/conv2d_6/Conv2DConv2Dinputs_0.model_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
model_2/conv2d_6/Conv2D?
'model_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_2/conv2d_6/BiasAdd/ReadVariableOp?
model_2/conv2d_6/BiasAddBiasAdd model_2/conv2d_6/Conv2D:output:0/model_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
model_2/conv2d_6/BiasAdd?
model_2/conv2d_6/ReluRelu!model_2/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
model_2/conv2d_6/Relu?
model_2/max_pooling2d_6/MaxPoolMaxPool#model_2/conv2d_6/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2!
model_2/max_pooling2d_6/MaxPool?
,model_2/batch_normalization_6/ReadVariableOpReadVariableOp5model_2_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model_2/batch_normalization_6/ReadVariableOp?
.model_2/batch_normalization_6/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.model_2/batch_normalization_6/ReadVariableOp_1?
=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02?
=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02A
?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
.model_2/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3(model_2/max_pooling2d_6/MaxPool:output:04model_2/batch_normalization_6/ReadVariableOp:value:06model_2/batch_normalization_6/ReadVariableOp_1:value:0Emodel_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<20
.model_2/batch_normalization_6/FusedBatchNormV3?
,model_2/batch_normalization_6/AssignNewValueAssignVariableOpFmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource;model_2/batch_normalization_6/FusedBatchNormV3:batch_mean:0>^model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02.
,model_2/batch_normalization_6/AssignNewValue?
.model_2/batch_normalization_6/AssignNewValue_1AssignVariableOpHmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource?model_2/batch_normalization_6/FusedBatchNormV3:batch_variance:0@^model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype020
.model_2/batch_normalization_6/AssignNewValue_1?
model_2/dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2!
model_2/dropout_6/dropout/Const?
model_2/dropout_6/dropout/MulMul2model_2/batch_normalization_6/FusedBatchNormV3:y:0(model_2/dropout_6/dropout/Const:output:0*
T0*1
_output_shapes
:???????????@2
model_2/dropout_6/dropout/Mul?
model_2/dropout_6/dropout/ShapeShape2model_2/batch_normalization_6/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2!
model_2/dropout_6/dropout/Shape?
6model_2/dropout_6/dropout/random_uniform/RandomUniformRandomUniform(model_2/dropout_6/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype0*

seedY28
6model_2/dropout_6/dropout/random_uniform/RandomUniform?
(model_2/dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(model_2/dropout_6/dropout/GreaterEqual/y?
&model_2/dropout_6/dropout/GreaterEqualGreaterEqual?model_2/dropout_6/dropout/random_uniform/RandomUniform:output:01model_2/dropout_6/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2(
&model_2/dropout_6/dropout/GreaterEqual?
model_2/dropout_6/dropout/CastCast*model_2/dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2 
model_2/dropout_6/dropout/Cast?
model_2/dropout_6/dropout/Mul_1Mul!model_2/dropout_6/dropout/Mul:z:0"model_2/dropout_6/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@2!
model_2/dropout_6/dropout/Mul_1?
&model_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02(
&model_2/conv2d_7/Conv2D/ReadVariableOp?
model_2/conv2d_7/Conv2DConv2D#model_2/dropout_6/dropout/Mul_1:z:0.model_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model_2/conv2d_7/Conv2D?
'model_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_2/conv2d_7/BiasAdd/ReadVariableOp?
model_2/conv2d_7/BiasAddBiasAdd model_2/conv2d_7/Conv2D:output:0/model_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
model_2/conv2d_7/BiasAdd?
model_2/conv2d_7/ReluRelu!model_2/conv2d_7/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
model_2/conv2d_7/Relu?
model_2/max_pooling2d_7/MaxPoolMaxPool#model_2/conv2d_7/Relu:activations:0*0
_output_shapes
:?????????@@?*
ksize
*
paddingVALID*
strides
2!
model_2/max_pooling2d_7/MaxPool?
,model_2/batch_normalization_7/ReadVariableOpReadVariableOp5model_2_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,model_2/batch_normalization_7/ReadVariableOp?
.model_2/batch_normalization_7/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_7/ReadVariableOp_1?
=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02A
?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
.model_2/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3(model_2/max_pooling2d_7/MaxPool:output:04model_2/batch_normalization_7/ReadVariableOp:value:06model_2/batch_normalization_7/ReadVariableOp_1:value:0Emodel_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<20
.model_2/batch_normalization_7/FusedBatchNormV3?
,model_2/batch_normalization_7/AssignNewValueAssignVariableOpFmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource;model_2/batch_normalization_7/FusedBatchNormV3:batch_mean:0>^model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02.
,model_2/batch_normalization_7/AssignNewValue?
.model_2/batch_normalization_7/AssignNewValue_1AssignVariableOpHmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource?model_2/batch_normalization_7/FusedBatchNormV3:batch_variance:0@^model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype020
.model_2/batch_normalization_7/AssignNewValue_1?
model_2/dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2!
model_2/dropout_7/dropout/Const?
model_2/dropout_7/dropout/MulMul2model_2/batch_normalization_7/FusedBatchNormV3:y:0(model_2/dropout_7/dropout/Const:output:0*
T0*0
_output_shapes
:?????????@@?2
model_2/dropout_7/dropout/Mul?
model_2/dropout_7/dropout/ShapeShape2model_2/batch_normalization_7/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2!
model_2/dropout_7/dropout/Shape?
6model_2/dropout_7/dropout/random_uniform/RandomUniformRandomUniform(model_2/dropout_7/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????@@?*
dtype0*

seedY*
seed228
6model_2/dropout_7/dropout/random_uniform/RandomUniform?
(model_2/dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(model_2/dropout_7/dropout/GreaterEqual/y?
&model_2/dropout_7/dropout/GreaterEqualGreaterEqual?model_2/dropout_7/dropout/random_uniform/RandomUniform:output:01model_2/dropout_7/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????@@?2(
&model_2/dropout_7/dropout/GreaterEqual?
model_2/dropout_7/dropout/CastCast*model_2/dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????@@?2 
model_2/dropout_7/dropout/Cast?
model_2/dropout_7/dropout/Mul_1Mul!model_2/dropout_7/dropout/Mul:z:0"model_2/dropout_7/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????@@?2!
model_2/dropout_7/dropout/Mul_1?
&model_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&model_2/conv2d_8/Conv2D/ReadVariableOp?
model_2/conv2d_8/Conv2DConv2D#model_2/dropout_7/dropout/Mul_1:z:0.model_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
model_2/conv2d_8/Conv2D?
'model_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_2/conv2d_8/BiasAdd/ReadVariableOp?
model_2/conv2d_8/BiasAddBiasAdd model_2/conv2d_8/Conv2D:output:0/model_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2
model_2/conv2d_8/BiasAdd?
model_2/conv2d_8/ReluRelu!model_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
model_2/conv2d_8/Relu?
model_2/max_pooling2d_8/MaxPoolMaxPool#model_2/conv2d_8/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2!
model_2/max_pooling2d_8/MaxPool?
,model_2/batch_normalization_8/ReadVariableOpReadVariableOp5model_2_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,model_2/batch_normalization_8/ReadVariableOp?
.model_2/batch_normalization_8/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_8/ReadVariableOp_1?
=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02A
?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
.model_2/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3(model_2/max_pooling2d_8/MaxPool:output:04model_2/batch_normalization_8/ReadVariableOp:value:06model_2/batch_normalization_8/ReadVariableOp_1:value:0Emodel_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<20
.model_2/batch_normalization_8/FusedBatchNormV3?
,model_2/batch_normalization_8/AssignNewValueAssignVariableOpFmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource;model_2/batch_normalization_8/FusedBatchNormV3:batch_mean:0>^model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02.
,model_2/batch_normalization_8/AssignNewValue?
.model_2/batch_normalization_8/AssignNewValue_1AssignVariableOpHmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource?model_2/batch_normalization_8/FusedBatchNormV3:batch_variance:0@^model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype020
.model_2/batch_normalization_8/AssignNewValue_1?
model_2/dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2!
model_2/dropout_8/dropout/Const?
model_2/dropout_8/dropout/MulMul2model_2/batch_normalization_8/FusedBatchNormV3:y:0(model_2/dropout_8/dropout/Const:output:0*
T0*0
_output_shapes
:?????????  ?2
model_2/dropout_8/dropout/Mul?
model_2/dropout_8/dropout/ShapeShape2model_2/batch_normalization_8/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2!
model_2/dropout_8/dropout/Shape?
6model_2/dropout_8/dropout/random_uniform/RandomUniformRandomUniform(model_2/dropout_8/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????  ?*
dtype0*

seedY*
seed228
6model_2/dropout_8/dropout/random_uniform/RandomUniform?
(model_2/dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(model_2/dropout_8/dropout/GreaterEqual/y?
&model_2/dropout_8/dropout/GreaterEqualGreaterEqual?model_2/dropout_8/dropout/random_uniform/RandomUniform:output:01model_2/dropout_8/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????  ?2(
&model_2/dropout_8/dropout/GreaterEqual?
model_2/dropout_8/dropout/CastCast*model_2/dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????  ?2 
model_2/dropout_8/dropout/Cast?
model_2/dropout_8/dropout/Mul_1Mul!model_2/dropout_8/dropout/Mul:z:0"model_2/dropout_8/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????  ?2!
model_2/dropout_8/dropout/Mul_1?
&model_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&model_2/conv2d_9/Conv2D/ReadVariableOp?
model_2/conv2d_9/Conv2DConv2D#model_2/dropout_8/dropout/Mul_1:z:0.model_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
model_2/conv2d_9/Conv2D?
'model_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_2/conv2d_9/BiasAdd/ReadVariableOp?
model_2/conv2d_9/BiasAddBiasAdd model_2/conv2d_9/Conv2D:output:0/model_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2
model_2/conv2d_9/BiasAdd?
model_2/conv2d_9/ReluRelu!model_2/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?2
model_2/conv2d_9/Relu?
,model_2/batch_normalization_9/ReadVariableOpReadVariableOp5model_2_batch_normalization_9_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,model_2/batch_normalization_9/ReadVariableOp?
.model_2/batch_normalization_9/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_9/ReadVariableOp_1?
=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02A
?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
.model_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3#model_2/conv2d_9/Relu:activations:04model_2/batch_normalization_9/ReadVariableOp:value:06model_2/batch_normalization_9/ReadVariableOp_1:value:0Emodel_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<20
.model_2/batch_normalization_9/FusedBatchNormV3?
,model_2/batch_normalization_9/AssignNewValueAssignVariableOpFmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource;model_2/batch_normalization_9/FusedBatchNormV3:batch_mean:0>^model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02.
,model_2/batch_normalization_9/AssignNewValue?
.model_2/batch_normalization_9/AssignNewValue_1AssignVariableOpHmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource?model_2/batch_normalization_9/FusedBatchNormV3:batch_variance:0@^model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype020
.model_2/batch_normalization_9/AssignNewValue_1?
model_2/max_pooling2d_9/MaxPoolMaxPool2model_2/batch_normalization_9/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2!
model_2/max_pooling2d_9/MaxPool?
model_2/dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2!
model_2/dropout_9/dropout/Const?
model_2/dropout_9/dropout/MulMul(model_2/max_pooling2d_9/MaxPool:output:0(model_2/dropout_9/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
model_2/dropout_9/dropout/Mul?
model_2/dropout_9/dropout/ShapeShape(model_2/max_pooling2d_9/MaxPool:output:0*
T0*
_output_shapes
:2!
model_2/dropout_9/dropout/Shape?
6model_2/dropout_9/dropout/random_uniform/RandomUniformRandomUniform(model_2/dropout_9/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0*

seedY*
seed228
6model_2/dropout_9/dropout/random_uniform/RandomUniform?
(model_2/dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(model_2/dropout_9/dropout/GreaterEqual/y?
&model_2/dropout_9/dropout/GreaterEqualGreaterEqual?model_2/dropout_9/dropout/random_uniform/RandomUniform:output:01model_2/dropout_9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2(
&model_2/dropout_9/dropout/GreaterEqual?
model_2/dropout_9/dropout/CastCast*model_2/dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2 
model_2/dropout_9/dropout/Cast?
model_2/dropout_9/dropout/Mul_1Mul!model_2/dropout_9/dropout/Mul:z:0"model_2/dropout_9/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2!
model_2/dropout_9/dropout/Mul_1?
9model_2/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9model_2/global_average_pooling2d_1/Mean/reduction_indices?
'model_2/global_average_pooling2d_1/MeanMean#model_2/dropout_9/dropout/Mul_1:z:0Bmodel_2/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2)
'model_2/global_average_pooling2d_1/Mean?
%model_2/dense_1/MatMul/ReadVariableOpReadVariableOp.model_2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%model_2/dense_1/MatMul/ReadVariableOp?
model_2/dense_1/MatMulMatMul0model_2/global_average_pooling2d_1/Mean:output:0-model_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense_1/MatMul?
&model_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&model_2/dense_1/BiasAdd/ReadVariableOp?
model_2/dense_1/BiasAddBiasAdd model_2/dense_1/MatMul:product:0.model_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense_1/BiasAdd?
(model_2/conv2d_6/Conv2D_1/ReadVariableOpReadVariableOp/model_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(model_2/conv2d_6/Conv2D_1/ReadVariableOp?
model_2/conv2d_6/Conv2D_1Conv2Dinputs_10model_2/conv2d_6/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
model_2/conv2d_6/Conv2D_1?
)model_2/conv2d_6/BiasAdd_1/ReadVariableOpReadVariableOp0model_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_2/conv2d_6/BiasAdd_1/ReadVariableOp?
model_2/conv2d_6/BiasAdd_1BiasAdd"model_2/conv2d_6/Conv2D_1:output:01model_2/conv2d_6/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
model_2/conv2d_6/BiasAdd_1?
model_2/conv2d_6/Relu_1Relu#model_2/conv2d_6/BiasAdd_1:output:0*
T0*1
_output_shapes
:???????????@2
model_2/conv2d_6/Relu_1?
!model_2/max_pooling2d_6/MaxPool_1MaxPool%model_2/conv2d_6/Relu_1:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2#
!model_2/max_pooling2d_6/MaxPool_1?
.model_2/batch_normalization_6/ReadVariableOp_2ReadVariableOp5model_2_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype020
.model_2/batch_normalization_6/ReadVariableOp_2?
.model_2/batch_normalization_6/ReadVariableOp_3ReadVariableOp7model_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.model_2/batch_normalization_6/ReadVariableOp_3?
?model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource-^model_2/batch_normalization_6/AssignNewValue*
_output_shapes
:@*
dtype02A
?model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp?
Amodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource/^model_2/batch_normalization_6/AssignNewValue_1*
_output_shapes
:@*
dtype02C
Amodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1?
0model_2/batch_normalization_6/FusedBatchNormV3_1FusedBatchNormV3*model_2/max_pooling2d_6/MaxPool_1:output:06model_2/batch_normalization_6/ReadVariableOp_2:value:06model_2/batch_normalization_6/ReadVariableOp_3:value:0Gmodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp:value:0Imodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<22
0model_2/batch_normalization_6/FusedBatchNormV3_1?
.model_2/batch_normalization_6/AssignNewValue_2AssignVariableOpFmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource=model_2/batch_normalization_6/FusedBatchNormV3_1:batch_mean:0-^model_2/batch_normalization_6/AssignNewValue@^model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype020
.model_2/batch_normalization_6/AssignNewValue_2?
.model_2/batch_normalization_6/AssignNewValue_3AssignVariableOpHmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceAmodel_2/batch_normalization_6/FusedBatchNormV3_1:batch_variance:0/^model_2/batch_normalization_6/AssignNewValue_1B^model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype020
.model_2/batch_normalization_6/AssignNewValue_3?
!model_2/dropout_6/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!model_2/dropout_6/dropout_1/Const?
model_2/dropout_6/dropout_1/MulMul4model_2/batch_normalization_6/FusedBatchNormV3_1:y:0*model_2/dropout_6/dropout_1/Const:output:0*
T0*1
_output_shapes
:???????????@2!
model_2/dropout_6/dropout_1/Mul?
!model_2/dropout_6/dropout_1/ShapeShape4model_2/batch_normalization_6/FusedBatchNormV3_1:y:0*
T0*
_output_shapes
:2#
!model_2/dropout_6/dropout_1/Shape?
8model_2/dropout_6/dropout_1/random_uniform/RandomUniformRandomUniform*model_2/dropout_6/dropout_1/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype0*

seedY*
seed22:
8model_2/dropout_6/dropout_1/random_uniform/RandomUniform?
*model_2/dropout_6/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*model_2/dropout_6/dropout_1/GreaterEqual/y?
(model_2/dropout_6/dropout_1/GreaterEqualGreaterEqualAmodel_2/dropout_6/dropout_1/random_uniform/RandomUniform:output:03model_2/dropout_6/dropout_1/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2*
(model_2/dropout_6/dropout_1/GreaterEqual?
 model_2/dropout_6/dropout_1/CastCast,model_2/dropout_6/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2"
 model_2/dropout_6/dropout_1/Cast?
!model_2/dropout_6/dropout_1/Mul_1Mul#model_2/dropout_6/dropout_1/Mul:z:0$model_2/dropout_6/dropout_1/Cast:y:0*
T0*1
_output_shapes
:???????????@2#
!model_2/dropout_6/dropout_1/Mul_1?
(model_2/conv2d_7/Conv2D_1/ReadVariableOpReadVariableOp/model_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02*
(model_2/conv2d_7/Conv2D_1/ReadVariableOp?
model_2/conv2d_7/Conv2D_1Conv2D%model_2/dropout_6/dropout_1/Mul_1:z:00model_2/conv2d_7/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model_2/conv2d_7/Conv2D_1?
)model_2/conv2d_7/BiasAdd_1/ReadVariableOpReadVariableOp0model_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_2/conv2d_7/BiasAdd_1/ReadVariableOp?
model_2/conv2d_7/BiasAdd_1BiasAdd"model_2/conv2d_7/Conv2D_1:output:01model_2/conv2d_7/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
model_2/conv2d_7/BiasAdd_1?
model_2/conv2d_7/Relu_1Relu#model_2/conv2d_7/BiasAdd_1:output:0*
T0*2
_output_shapes 
:????????????2
model_2/conv2d_7/Relu_1?
!model_2/max_pooling2d_7/MaxPool_1MaxPool%model_2/conv2d_7/Relu_1:activations:0*0
_output_shapes
:?????????@@?*
ksize
*
paddingVALID*
strides
2#
!model_2/max_pooling2d_7/MaxPool_1?
.model_2/batch_normalization_7/ReadVariableOp_2ReadVariableOp5model_2_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_7/ReadVariableOp_2?
.model_2/batch_normalization_7/ReadVariableOp_3ReadVariableOp7model_2_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_7/ReadVariableOp_3?
?model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource-^model_2/batch_normalization_7/AssignNewValue*
_output_shapes	
:?*
dtype02A
?model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp?
Amodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource/^model_2/batch_normalization_7/AssignNewValue_1*
_output_shapes	
:?*
dtype02C
Amodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1?
0model_2/batch_normalization_7/FusedBatchNormV3_1FusedBatchNormV3*model_2/max_pooling2d_7/MaxPool_1:output:06model_2/batch_normalization_7/ReadVariableOp_2:value:06model_2/batch_normalization_7/ReadVariableOp_3:value:0Gmodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp:value:0Imodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<22
0model_2/batch_normalization_7/FusedBatchNormV3_1?
.model_2/batch_normalization_7/AssignNewValue_2AssignVariableOpFmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource=model_2/batch_normalization_7/FusedBatchNormV3_1:batch_mean:0-^model_2/batch_normalization_7/AssignNewValue@^model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype020
.model_2/batch_normalization_7/AssignNewValue_2?
.model_2/batch_normalization_7/AssignNewValue_3AssignVariableOpHmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceAmodel_2/batch_normalization_7/FusedBatchNormV3_1:batch_variance:0/^model_2/batch_normalization_7/AssignNewValue_1B^model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype020
.model_2/batch_normalization_7/AssignNewValue_3?
!model_2/dropout_7/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!model_2/dropout_7/dropout_1/Const?
model_2/dropout_7/dropout_1/MulMul4model_2/batch_normalization_7/FusedBatchNormV3_1:y:0*model_2/dropout_7/dropout_1/Const:output:0*
T0*0
_output_shapes
:?????????@@?2!
model_2/dropout_7/dropout_1/Mul?
!model_2/dropout_7/dropout_1/ShapeShape4model_2/batch_normalization_7/FusedBatchNormV3_1:y:0*
T0*
_output_shapes
:2#
!model_2/dropout_7/dropout_1/Shape?
8model_2/dropout_7/dropout_1/random_uniform/RandomUniformRandomUniform*model_2/dropout_7/dropout_1/Shape:output:0*
T0*0
_output_shapes
:?????????@@?*
dtype0*

seedY*
seed22:
8model_2/dropout_7/dropout_1/random_uniform/RandomUniform?
*model_2/dropout_7/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*model_2/dropout_7/dropout_1/GreaterEqual/y?
(model_2/dropout_7/dropout_1/GreaterEqualGreaterEqualAmodel_2/dropout_7/dropout_1/random_uniform/RandomUniform:output:03model_2/dropout_7/dropout_1/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????@@?2*
(model_2/dropout_7/dropout_1/GreaterEqual?
 model_2/dropout_7/dropout_1/CastCast,model_2/dropout_7/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????@@?2"
 model_2/dropout_7/dropout_1/Cast?
!model_2/dropout_7/dropout_1/Mul_1Mul#model_2/dropout_7/dropout_1/Mul:z:0$model_2/dropout_7/dropout_1/Cast:y:0*
T0*0
_output_shapes
:?????????@@?2#
!model_2/dropout_7/dropout_1/Mul_1?
(model_2/conv2d_8/Conv2D_1/ReadVariableOpReadVariableOp/model_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model_2/conv2d_8/Conv2D_1/ReadVariableOp?
model_2/conv2d_8/Conv2D_1Conv2D%model_2/dropout_7/dropout_1/Mul_1:z:00model_2/conv2d_8/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
model_2/conv2d_8/Conv2D_1?
)model_2/conv2d_8/BiasAdd_1/ReadVariableOpReadVariableOp0model_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_2/conv2d_8/BiasAdd_1/ReadVariableOp?
model_2/conv2d_8/BiasAdd_1BiasAdd"model_2/conv2d_8/Conv2D_1:output:01model_2/conv2d_8/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2
model_2/conv2d_8/BiasAdd_1?
model_2/conv2d_8/Relu_1Relu#model_2/conv2d_8/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????@@?2
model_2/conv2d_8/Relu_1?
!model_2/max_pooling2d_8/MaxPool_1MaxPool%model_2/conv2d_8/Relu_1:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2#
!model_2/max_pooling2d_8/MaxPool_1?
.model_2/batch_normalization_8/ReadVariableOp_2ReadVariableOp5model_2_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_8/ReadVariableOp_2?
.model_2/batch_normalization_8/ReadVariableOp_3ReadVariableOp7model_2_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_8/ReadVariableOp_3?
?model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource-^model_2/batch_normalization_8/AssignNewValue*
_output_shapes	
:?*
dtype02A
?model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp?
Amodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource/^model_2/batch_normalization_8/AssignNewValue_1*
_output_shapes	
:?*
dtype02C
Amodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1?
0model_2/batch_normalization_8/FusedBatchNormV3_1FusedBatchNormV3*model_2/max_pooling2d_8/MaxPool_1:output:06model_2/batch_normalization_8/ReadVariableOp_2:value:06model_2/batch_normalization_8/ReadVariableOp_3:value:0Gmodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp:value:0Imodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<22
0model_2/batch_normalization_8/FusedBatchNormV3_1?
.model_2/batch_normalization_8/AssignNewValue_2AssignVariableOpFmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource=model_2/batch_normalization_8/FusedBatchNormV3_1:batch_mean:0-^model_2/batch_normalization_8/AssignNewValue@^model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype020
.model_2/batch_normalization_8/AssignNewValue_2?
.model_2/batch_normalization_8/AssignNewValue_3AssignVariableOpHmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceAmodel_2/batch_normalization_8/FusedBatchNormV3_1:batch_variance:0/^model_2/batch_normalization_8/AssignNewValue_1B^model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype020
.model_2/batch_normalization_8/AssignNewValue_3?
!model_2/dropout_8/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!model_2/dropout_8/dropout_1/Const?
model_2/dropout_8/dropout_1/MulMul4model_2/batch_normalization_8/FusedBatchNormV3_1:y:0*model_2/dropout_8/dropout_1/Const:output:0*
T0*0
_output_shapes
:?????????  ?2!
model_2/dropout_8/dropout_1/Mul?
!model_2/dropout_8/dropout_1/ShapeShape4model_2/batch_normalization_8/FusedBatchNormV3_1:y:0*
T0*
_output_shapes
:2#
!model_2/dropout_8/dropout_1/Shape?
8model_2/dropout_8/dropout_1/random_uniform/RandomUniformRandomUniform*model_2/dropout_8/dropout_1/Shape:output:0*
T0*0
_output_shapes
:?????????  ?*
dtype0*

seedY*
seed22:
8model_2/dropout_8/dropout_1/random_uniform/RandomUniform?
*model_2/dropout_8/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*model_2/dropout_8/dropout_1/GreaterEqual/y?
(model_2/dropout_8/dropout_1/GreaterEqualGreaterEqualAmodel_2/dropout_8/dropout_1/random_uniform/RandomUniform:output:03model_2/dropout_8/dropout_1/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????  ?2*
(model_2/dropout_8/dropout_1/GreaterEqual?
 model_2/dropout_8/dropout_1/CastCast,model_2/dropout_8/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????  ?2"
 model_2/dropout_8/dropout_1/Cast?
!model_2/dropout_8/dropout_1/Mul_1Mul#model_2/dropout_8/dropout_1/Mul:z:0$model_2/dropout_8/dropout_1/Cast:y:0*
T0*0
_output_shapes
:?????????  ?2#
!model_2/dropout_8/dropout_1/Mul_1?
(model_2/conv2d_9/Conv2D_1/ReadVariableOpReadVariableOp/model_2_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model_2/conv2d_9/Conv2D_1/ReadVariableOp?
model_2/conv2d_9/Conv2D_1Conv2D%model_2/dropout_8/dropout_1/Mul_1:z:00model_2/conv2d_9/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
model_2/conv2d_9/Conv2D_1?
)model_2/conv2d_9/BiasAdd_1/ReadVariableOpReadVariableOp0model_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_2/conv2d_9/BiasAdd_1/ReadVariableOp?
model_2/conv2d_9/BiasAdd_1BiasAdd"model_2/conv2d_9/Conv2D_1:output:01model_2/conv2d_9/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2
model_2/conv2d_9/BiasAdd_1?
model_2/conv2d_9/Relu_1Relu#model_2/conv2d_9/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????  ?2
model_2/conv2d_9/Relu_1?
.model_2/batch_normalization_9/ReadVariableOp_2ReadVariableOp5model_2_batch_normalization_9_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_9/ReadVariableOp_2?
.model_2/batch_normalization_9/ReadVariableOp_3ReadVariableOp7model_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_9/ReadVariableOp_3?
?model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource-^model_2/batch_normalization_9/AssignNewValue*
_output_shapes	
:?*
dtype02A
?model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp?
Amodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource/^model_2/batch_normalization_9/AssignNewValue_1*
_output_shapes	
:?*
dtype02C
Amodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1?
0model_2/batch_normalization_9/FusedBatchNormV3_1FusedBatchNormV3%model_2/conv2d_9/Relu_1:activations:06model_2/batch_normalization_9/ReadVariableOp_2:value:06model_2/batch_normalization_9/ReadVariableOp_3:value:0Gmodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp:value:0Imodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<22
0model_2/batch_normalization_9/FusedBatchNormV3_1?
.model_2/batch_normalization_9/AssignNewValue_2AssignVariableOpFmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource=model_2/batch_normalization_9/FusedBatchNormV3_1:batch_mean:0-^model_2/batch_normalization_9/AssignNewValue@^model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp*
_output_shapes
 *
dtype020
.model_2/batch_normalization_9/AssignNewValue_2?
.model_2/batch_normalization_9/AssignNewValue_3AssignVariableOpHmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceAmodel_2/batch_normalization_9/FusedBatchNormV3_1:batch_variance:0/^model_2/batch_normalization_9/AssignNewValue_1B^model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1*
_output_shapes
 *
dtype020
.model_2/batch_normalization_9/AssignNewValue_3?
!model_2/max_pooling2d_9/MaxPool_1MaxPool4model_2/batch_normalization_9/FusedBatchNormV3_1:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2#
!model_2/max_pooling2d_9/MaxPool_1?
!model_2/dropout_9/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2#
!model_2/dropout_9/dropout_1/Const?
model_2/dropout_9/dropout_1/MulMul*model_2/max_pooling2d_9/MaxPool_1:output:0*model_2/dropout_9/dropout_1/Const:output:0*
T0*0
_output_shapes
:??????????2!
model_2/dropout_9/dropout_1/Mul?
!model_2/dropout_9/dropout_1/ShapeShape*model_2/max_pooling2d_9/MaxPool_1:output:0*
T0*
_output_shapes
:2#
!model_2/dropout_9/dropout_1/Shape?
8model_2/dropout_9/dropout_1/random_uniform/RandomUniformRandomUniform*model_2/dropout_9/dropout_1/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0*

seedY*
seed22:
8model_2/dropout_9/dropout_1/random_uniform/RandomUniform?
*model_2/dropout_9/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2,
*model_2/dropout_9/dropout_1/GreaterEqual/y?
(model_2/dropout_9/dropout_1/GreaterEqualGreaterEqualAmodel_2/dropout_9/dropout_1/random_uniform/RandomUniform:output:03model_2/dropout_9/dropout_1/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2*
(model_2/dropout_9/dropout_1/GreaterEqual?
 model_2/dropout_9/dropout_1/CastCast,model_2/dropout_9/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2"
 model_2/dropout_9/dropout_1/Cast?
!model_2/dropout_9/dropout_1/Mul_1Mul#model_2/dropout_9/dropout_1/Mul:z:0$model_2/dropout_9/dropout_1/Cast:y:0*
T0*0
_output_shapes
:??????????2#
!model_2/dropout_9/dropout_1/Mul_1?
;model_2/global_average_pooling2d_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;model_2/global_average_pooling2d_1/Mean_1/reduction_indices?
)model_2/global_average_pooling2d_1/Mean_1Mean%model_2/dropout_9/dropout_1/Mul_1:z:0Dmodel_2/global_average_pooling2d_1/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2+
)model_2/global_average_pooling2d_1/Mean_1?
'model_2/dense_1/MatMul_1/ReadVariableOpReadVariableOp.model_2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'model_2/dense_1/MatMul_1/ReadVariableOp?
model_2/dense_1/MatMul_1MatMul2model_2/global_average_pooling2d_1/Mean_1:output:0/model_2/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense_1/MatMul_1?
(model_2/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp/model_2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_2/dense_1/BiasAdd_1/ReadVariableOp?
model_2/dense_1/BiasAdd_1BiasAdd"model_2/dense_1/MatMul_1:product:00model_2/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense_1/BiasAdd_1?
lambda_1/subSub model_2/dense_1/BiasAdd:output:0"model_2/dense_1/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
lambda_1/subq
lambda_1/SquareSquarelambda_1/sub:z:0*
T0*(
_output_shapes
:??????????2
lambda_1/Square?
lambda_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
lambda_1/Sum/reduction_indices?
lambda_1/SumSumlambda_1/Square:y:0'lambda_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
lambda_1/Summ
lambda_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
lambda_1/Maximum/y?
lambda_1/MaximumMaximumlambda_1/Sum:output:0lambda_1/Maximum/y:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/Maximume
lambda_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda_1/Const?
lambda_1/Maximum_1Maximumlambda_1/Maximum:z:0lambda_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/Maximum_1p
lambda_1/SqrtSqrtlambda_1/Maximum_1:z:0*
T0*'
_output_shapes
:?????????2
lambda_1/Sqrtl
IdentityIdentitylambda_1/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp-^model_2/batch_normalization_6/AssignNewValue/^model_2/batch_normalization_6/AssignNewValue_1/^model_2/batch_normalization_6/AssignNewValue_2/^model_2/batch_normalization_6/AssignNewValue_3>^model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1@^model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpB^model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1-^model_2/batch_normalization_6/ReadVariableOp/^model_2/batch_normalization_6/ReadVariableOp_1/^model_2/batch_normalization_6/ReadVariableOp_2/^model_2/batch_normalization_6/ReadVariableOp_3-^model_2/batch_normalization_7/AssignNewValue/^model_2/batch_normalization_7/AssignNewValue_1/^model_2/batch_normalization_7/AssignNewValue_2/^model_2/batch_normalization_7/AssignNewValue_3>^model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1@^model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpB^model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1-^model_2/batch_normalization_7/ReadVariableOp/^model_2/batch_normalization_7/ReadVariableOp_1/^model_2/batch_normalization_7/ReadVariableOp_2/^model_2/batch_normalization_7/ReadVariableOp_3-^model_2/batch_normalization_8/AssignNewValue/^model_2/batch_normalization_8/AssignNewValue_1/^model_2/batch_normalization_8/AssignNewValue_2/^model_2/batch_normalization_8/AssignNewValue_3>^model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1@^model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOpB^model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1-^model_2/batch_normalization_8/ReadVariableOp/^model_2/batch_normalization_8/ReadVariableOp_1/^model_2/batch_normalization_8/ReadVariableOp_2/^model_2/batch_normalization_8/ReadVariableOp_3-^model_2/batch_normalization_9/AssignNewValue/^model_2/batch_normalization_9/AssignNewValue_1/^model_2/batch_normalization_9/AssignNewValue_2/^model_2/batch_normalization_9/AssignNewValue_3>^model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1@^model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOpB^model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1-^model_2/batch_normalization_9/ReadVariableOp/^model_2/batch_normalization_9/ReadVariableOp_1/^model_2/batch_normalization_9/ReadVariableOp_2/^model_2/batch_normalization_9/ReadVariableOp_3(^model_2/conv2d_6/BiasAdd/ReadVariableOp*^model_2/conv2d_6/BiasAdd_1/ReadVariableOp'^model_2/conv2d_6/Conv2D/ReadVariableOp)^model_2/conv2d_6/Conv2D_1/ReadVariableOp(^model_2/conv2d_7/BiasAdd/ReadVariableOp*^model_2/conv2d_7/BiasAdd_1/ReadVariableOp'^model_2/conv2d_7/Conv2D/ReadVariableOp)^model_2/conv2d_7/Conv2D_1/ReadVariableOp(^model_2/conv2d_8/BiasAdd/ReadVariableOp*^model_2/conv2d_8/BiasAdd_1/ReadVariableOp'^model_2/conv2d_8/Conv2D/ReadVariableOp)^model_2/conv2d_8/Conv2D_1/ReadVariableOp(^model_2/conv2d_9/BiasAdd/ReadVariableOp*^model_2/conv2d_9/BiasAdd_1/ReadVariableOp'^model_2/conv2d_9/Conv2D/ReadVariableOp)^model_2/conv2d_9/Conv2D_1/ReadVariableOp'^model_2/dense_1/BiasAdd/ReadVariableOp)^model_2/dense_1/BiasAdd_1/ReadVariableOp&^model_2/dense_1/MatMul/ReadVariableOp(^model_2/dense_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,model_2/batch_normalization_6/AssignNewValue,model_2/batch_normalization_6/AssignNewValue2`
.model_2/batch_normalization_6/AssignNewValue_1.model_2/batch_normalization_6/AssignNewValue_12`
.model_2/batch_normalization_6/AssignNewValue_2.model_2/batch_normalization_6/AssignNewValue_22`
.model_2/batch_normalization_6/AssignNewValue_3.model_2/batch_normalization_6/AssignNewValue_32~
=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12?
?model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp?model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp2?
Amodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1Amodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_12\
,model_2/batch_normalization_6/ReadVariableOp,model_2/batch_normalization_6/ReadVariableOp2`
.model_2/batch_normalization_6/ReadVariableOp_1.model_2/batch_normalization_6/ReadVariableOp_12`
.model_2/batch_normalization_6/ReadVariableOp_2.model_2/batch_normalization_6/ReadVariableOp_22`
.model_2/batch_normalization_6/ReadVariableOp_3.model_2/batch_normalization_6/ReadVariableOp_32\
,model_2/batch_normalization_7/AssignNewValue,model_2/batch_normalization_7/AssignNewValue2`
.model_2/batch_normalization_7/AssignNewValue_1.model_2/batch_normalization_7/AssignNewValue_12`
.model_2/batch_normalization_7/AssignNewValue_2.model_2/batch_normalization_7/AssignNewValue_22`
.model_2/batch_normalization_7/AssignNewValue_3.model_2/batch_normalization_7/AssignNewValue_32~
=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12?
?model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp?model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp2?
Amodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1Amodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_12\
,model_2/batch_normalization_7/ReadVariableOp,model_2/batch_normalization_7/ReadVariableOp2`
.model_2/batch_normalization_7/ReadVariableOp_1.model_2/batch_normalization_7/ReadVariableOp_12`
.model_2/batch_normalization_7/ReadVariableOp_2.model_2/batch_normalization_7/ReadVariableOp_22`
.model_2/batch_normalization_7/ReadVariableOp_3.model_2/batch_normalization_7/ReadVariableOp_32\
,model_2/batch_normalization_8/AssignNewValue,model_2/batch_normalization_8/AssignNewValue2`
.model_2/batch_normalization_8/AssignNewValue_1.model_2/batch_normalization_8/AssignNewValue_12`
.model_2/batch_normalization_8/AssignNewValue_2.model_2/batch_normalization_8/AssignNewValue_22`
.model_2/batch_normalization_8/AssignNewValue_3.model_2/batch_normalization_8/AssignNewValue_32~
=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12?
?model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp?model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp2?
Amodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1Amodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_12\
,model_2/batch_normalization_8/ReadVariableOp,model_2/batch_normalization_8/ReadVariableOp2`
.model_2/batch_normalization_8/ReadVariableOp_1.model_2/batch_normalization_8/ReadVariableOp_12`
.model_2/batch_normalization_8/ReadVariableOp_2.model_2/batch_normalization_8/ReadVariableOp_22`
.model_2/batch_normalization_8/ReadVariableOp_3.model_2/batch_normalization_8/ReadVariableOp_32\
,model_2/batch_normalization_9/AssignNewValue,model_2/batch_normalization_9/AssignNewValue2`
.model_2/batch_normalization_9/AssignNewValue_1.model_2/batch_normalization_9/AssignNewValue_12`
.model_2/batch_normalization_9/AssignNewValue_2.model_2/batch_normalization_9/AssignNewValue_22`
.model_2/batch_normalization_9/AssignNewValue_3.model_2/batch_normalization_9/AssignNewValue_32~
=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12?
?model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp?model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp2?
Amodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1Amodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_12\
,model_2/batch_normalization_9/ReadVariableOp,model_2/batch_normalization_9/ReadVariableOp2`
.model_2/batch_normalization_9/ReadVariableOp_1.model_2/batch_normalization_9/ReadVariableOp_12`
.model_2/batch_normalization_9/ReadVariableOp_2.model_2/batch_normalization_9/ReadVariableOp_22`
.model_2/batch_normalization_9/ReadVariableOp_3.model_2/batch_normalization_9/ReadVariableOp_32R
'model_2/conv2d_6/BiasAdd/ReadVariableOp'model_2/conv2d_6/BiasAdd/ReadVariableOp2V
)model_2/conv2d_6/BiasAdd_1/ReadVariableOp)model_2/conv2d_6/BiasAdd_1/ReadVariableOp2P
&model_2/conv2d_6/Conv2D/ReadVariableOp&model_2/conv2d_6/Conv2D/ReadVariableOp2T
(model_2/conv2d_6/Conv2D_1/ReadVariableOp(model_2/conv2d_6/Conv2D_1/ReadVariableOp2R
'model_2/conv2d_7/BiasAdd/ReadVariableOp'model_2/conv2d_7/BiasAdd/ReadVariableOp2V
)model_2/conv2d_7/BiasAdd_1/ReadVariableOp)model_2/conv2d_7/BiasAdd_1/ReadVariableOp2P
&model_2/conv2d_7/Conv2D/ReadVariableOp&model_2/conv2d_7/Conv2D/ReadVariableOp2T
(model_2/conv2d_7/Conv2D_1/ReadVariableOp(model_2/conv2d_7/Conv2D_1/ReadVariableOp2R
'model_2/conv2d_8/BiasAdd/ReadVariableOp'model_2/conv2d_8/BiasAdd/ReadVariableOp2V
)model_2/conv2d_8/BiasAdd_1/ReadVariableOp)model_2/conv2d_8/BiasAdd_1/ReadVariableOp2P
&model_2/conv2d_8/Conv2D/ReadVariableOp&model_2/conv2d_8/Conv2D/ReadVariableOp2T
(model_2/conv2d_8/Conv2D_1/ReadVariableOp(model_2/conv2d_8/Conv2D_1/ReadVariableOp2R
'model_2/conv2d_9/BiasAdd/ReadVariableOp'model_2/conv2d_9/BiasAdd/ReadVariableOp2V
)model_2/conv2d_9/BiasAdd_1/ReadVariableOp)model_2/conv2d_9/BiasAdd_1/ReadVariableOp2P
&model_2/conv2d_9/Conv2D/ReadVariableOp&model_2/conv2d_9/Conv2D/ReadVariableOp2T
(model_2/conv2d_9/Conv2D_1/ReadVariableOp(model_2/conv2d_9/Conv2D_1/ReadVariableOp2P
&model_2/dense_1/BiasAdd/ReadVariableOp&model_2/dense_1/BiasAdd/ReadVariableOp2T
(model_2/dense_1/BiasAdd_1/ReadVariableOp(model_2/dense_1/BiasAdd_1/ReadVariableOp2N
%model_2/dense_1/MatMul/ReadVariableOp%model_2/dense_1/MatMul/ReadVariableOp2R
'model_2/dense_1/MatMul_1/ReadVariableOp'model_2/dense_1/MatMul_1/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
K
/__inference_max_pooling2d_7_layer_call_fn_69355

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_665352
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69373

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_68186
input_4
input_5!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:
??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4input_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *)
f$R"
 __inference__wrapped_model_658342
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_5
?	
?
5__inference_batch_normalization_8_layer_call_fn_69657

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_666112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
o
C__inference_lambda_1_layer_call_and_return_conditional_losses_69112
inputs_0
inputs_1
identityX
subSubinputs_0inputs_1*
T0*(
_output_shapes
:??????????2
subV
SquareSquaresub:z:0*
T0*(
_output_shapes
:??????????2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Sum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
	Maximum/yq
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constp
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
	Maximum_1U
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
V
:__inference_global_average_pooling2d_1_layer_call_fn_69905

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *^
fYRW
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_664362
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_69536

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_69687

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????  ?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????  ?*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????  ?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????  ?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????  ?2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_69340

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_66626

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????  ?2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????  ?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_66026

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_66963

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????@@?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????@@?*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????@@?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????@@?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????@@?2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
T
(__inference_lambda_1_layer_call_fn_69124
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_lambda_1_layer_call_and_return_conditional_losses_676852
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
??
?)
 __inference__wrapped_model_65834
input_4
input_5Q
7model_3_model_2_conv2d_6_conv2d_readvariableop_resource:@F
8model_3_model_2_conv2d_6_biasadd_readvariableop_resource:@K
=model_3_model_2_batch_normalization_6_readvariableop_resource:@M
?model_3_model_2_batch_normalization_6_readvariableop_1_resource:@\
Nmodel_3_model_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@^
Pmodel_3_model_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@R
7model_3_model_2_conv2d_7_conv2d_readvariableop_resource:@?G
8model_3_model_2_conv2d_7_biasadd_readvariableop_resource:	?L
=model_3_model_2_batch_normalization_7_readvariableop_resource:	?N
?model_3_model_2_batch_normalization_7_readvariableop_1_resource:	?]
Nmodel_3_model_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	?_
Pmodel_3_model_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	?S
7model_3_model_2_conv2d_8_conv2d_readvariableop_resource:??G
8model_3_model_2_conv2d_8_biasadd_readvariableop_resource:	?L
=model_3_model_2_batch_normalization_8_readvariableop_resource:	?N
?model_3_model_2_batch_normalization_8_readvariableop_1_resource:	?]
Nmodel_3_model_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	?_
Pmodel_3_model_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	?S
7model_3_model_2_conv2d_9_conv2d_readvariableop_resource:??G
8model_3_model_2_conv2d_9_biasadd_readvariableop_resource:	?L
=model_3_model_2_batch_normalization_9_readvariableop_resource:	?N
?model_3_model_2_batch_normalization_9_readvariableop_1_resource:	?]
Nmodel_3_model_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	?_
Pmodel_3_model_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	?J
6model_3_model_2_dense_1_matmul_readvariableop_resource:
??F
7model_3_model_2_dense_1_biasadd_readvariableop_resource:	?
identity??Emodel_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?Gmodel_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?Gmodel_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp?Imodel_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1?4model_3/model_2/batch_normalization_6/ReadVariableOp?6model_3/model_2/batch_normalization_6/ReadVariableOp_1?6model_3/model_2/batch_normalization_6/ReadVariableOp_2?6model_3/model_2/batch_normalization_6/ReadVariableOp_3?Emodel_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?Gmodel_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?Gmodel_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp?Imodel_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1?4model_3/model_2/batch_normalization_7/ReadVariableOp?6model_3/model_2/batch_normalization_7/ReadVariableOp_1?6model_3/model_2/batch_normalization_7/ReadVariableOp_2?6model_3/model_2/batch_normalization_7/ReadVariableOp_3?Emodel_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?Gmodel_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?Gmodel_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp?Imodel_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1?4model_3/model_2/batch_normalization_8/ReadVariableOp?6model_3/model_2/batch_normalization_8/ReadVariableOp_1?6model_3/model_2/batch_normalization_8/ReadVariableOp_2?6model_3/model_2/batch_normalization_8/ReadVariableOp_3?Emodel_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?Gmodel_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?Gmodel_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp?Imodel_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1?4model_3/model_2/batch_normalization_9/ReadVariableOp?6model_3/model_2/batch_normalization_9/ReadVariableOp_1?6model_3/model_2/batch_normalization_9/ReadVariableOp_2?6model_3/model_2/batch_normalization_9/ReadVariableOp_3?/model_3/model_2/conv2d_6/BiasAdd/ReadVariableOp?1model_3/model_2/conv2d_6/BiasAdd_1/ReadVariableOp?.model_3/model_2/conv2d_6/Conv2D/ReadVariableOp?0model_3/model_2/conv2d_6/Conv2D_1/ReadVariableOp?/model_3/model_2/conv2d_7/BiasAdd/ReadVariableOp?1model_3/model_2/conv2d_7/BiasAdd_1/ReadVariableOp?.model_3/model_2/conv2d_7/Conv2D/ReadVariableOp?0model_3/model_2/conv2d_7/Conv2D_1/ReadVariableOp?/model_3/model_2/conv2d_8/BiasAdd/ReadVariableOp?1model_3/model_2/conv2d_8/BiasAdd_1/ReadVariableOp?.model_3/model_2/conv2d_8/Conv2D/ReadVariableOp?0model_3/model_2/conv2d_8/Conv2D_1/ReadVariableOp?/model_3/model_2/conv2d_9/BiasAdd/ReadVariableOp?1model_3/model_2/conv2d_9/BiasAdd_1/ReadVariableOp?.model_3/model_2/conv2d_9/Conv2D/ReadVariableOp?0model_3/model_2/conv2d_9/Conv2D_1/ReadVariableOp?.model_3/model_2/dense_1/BiasAdd/ReadVariableOp?0model_3/model_2/dense_1/BiasAdd_1/ReadVariableOp?-model_3/model_2/dense_1/MatMul/ReadVariableOp?/model_3/model_2/dense_1/MatMul_1/ReadVariableOp?
.model_3/model_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp7model_3_model_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype020
.model_3/model_2/conv2d_6/Conv2D/ReadVariableOp?
model_3/model_2/conv2d_6/Conv2DConv2Dinput_46model_3/model_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2!
model_3/model_2/conv2d_6/Conv2D?
/model_3/model_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp8model_3_model_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/model_3/model_2/conv2d_6/BiasAdd/ReadVariableOp?
 model_3/model_2/conv2d_6/BiasAddBiasAdd(model_3/model_2/conv2d_6/Conv2D:output:07model_3/model_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2"
 model_3/model_2/conv2d_6/BiasAdd?
model_3/model_2/conv2d_6/ReluRelu)model_3/model_2/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
model_3/model_2/conv2d_6/Relu?
'model_3/model_2/max_pooling2d_6/MaxPoolMaxPool+model_3/model_2/conv2d_6/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2)
'model_3/model_2/max_pooling2d_6/MaxPool?
4model_3/model_2/batch_normalization_6/ReadVariableOpReadVariableOp=model_3_model_2_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype026
4model_3/model_2/batch_normalization_6/ReadVariableOp?
6model_3/model_2/batch_normalization_6/ReadVariableOp_1ReadVariableOp?model_3_model_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6model_3/model_2/batch_normalization_6/ReadVariableOp_1?
Emodel_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpNmodel_3_model_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02G
Emodel_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
Gmodel_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPmodel_3_model_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02I
Gmodel_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
6model_3/model_2/batch_normalization_6/FusedBatchNormV3FusedBatchNormV30model_3/model_2/max_pooling2d_6/MaxPool:output:0<model_3/model_2/batch_normalization_6/ReadVariableOp:value:0>model_3/model_2/batch_normalization_6/ReadVariableOp_1:value:0Mmodel_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Omodel_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 28
6model_3/model_2/batch_normalization_6/FusedBatchNormV3?
"model_3/model_2/dropout_6/IdentityIdentity:model_3/model_2/batch_normalization_6/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2$
"model_3/model_2/dropout_6/Identity?
.model_3/model_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp7model_3_model_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype020
.model_3/model_2/conv2d_7/Conv2D/ReadVariableOp?
model_3/model_2/conv2d_7/Conv2DConv2D+model_3/model_2/dropout_6/Identity:output:06model_3/model_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2!
model_3/model_2/conv2d_7/Conv2D?
/model_3/model_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp8model_3_model_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model_3/model_2/conv2d_7/BiasAdd/ReadVariableOp?
 model_3/model_2/conv2d_7/BiasAddBiasAdd(model_3/model_2/conv2d_7/Conv2D:output:07model_3/model_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2"
 model_3/model_2/conv2d_7/BiasAdd?
model_3/model_2/conv2d_7/ReluRelu)model_3/model_2/conv2d_7/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
model_3/model_2/conv2d_7/Relu?
'model_3/model_2/max_pooling2d_7/MaxPoolMaxPool+model_3/model_2/conv2d_7/Relu:activations:0*0
_output_shapes
:?????????@@?*
ksize
*
paddingVALID*
strides
2)
'model_3/model_2/max_pooling2d_7/MaxPool?
4model_3/model_2/batch_normalization_7/ReadVariableOpReadVariableOp=model_3_model_2_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype026
4model_3/model_2/batch_normalization_7/ReadVariableOp?
6model_3/model_2/batch_normalization_7/ReadVariableOp_1ReadVariableOp?model_3_model_2_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype028
6model_3/model_2/batch_normalization_7/ReadVariableOp_1?
Emodel_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpNmodel_3_model_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02G
Emodel_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
Gmodel_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPmodel_3_model_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02I
Gmodel_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
6model_3/model_2/batch_normalization_7/FusedBatchNormV3FusedBatchNormV30model_3/model_2/max_pooling2d_7/MaxPool:output:0<model_3/model_2/batch_normalization_7/ReadVariableOp:value:0>model_3/model_2/batch_normalization_7/ReadVariableOp_1:value:0Mmodel_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Omodel_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
is_training( 28
6model_3/model_2/batch_normalization_7/FusedBatchNormV3?
"model_3/model_2/dropout_7/IdentityIdentity:model_3/model_2/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@@?2$
"model_3/model_2/dropout_7/Identity?
.model_3/model_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp7model_3_model_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype020
.model_3/model_2/conv2d_8/Conv2D/ReadVariableOp?
model_3/model_2/conv2d_8/Conv2DConv2D+model_3/model_2/dropout_7/Identity:output:06model_3/model_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2!
model_3/model_2/conv2d_8/Conv2D?
/model_3/model_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp8model_3_model_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model_3/model_2/conv2d_8/BiasAdd/ReadVariableOp?
 model_3/model_2/conv2d_8/BiasAddBiasAdd(model_3/model_2/conv2d_8/Conv2D:output:07model_3/model_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2"
 model_3/model_2/conv2d_8/BiasAdd?
model_3/model_2/conv2d_8/ReluRelu)model_3/model_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
model_3/model_2/conv2d_8/Relu?
'model_3/model_2/max_pooling2d_8/MaxPoolMaxPool+model_3/model_2/conv2d_8/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2)
'model_3/model_2/max_pooling2d_8/MaxPool?
4model_3/model_2/batch_normalization_8/ReadVariableOpReadVariableOp=model_3_model_2_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype026
4model_3/model_2/batch_normalization_8/ReadVariableOp?
6model_3/model_2/batch_normalization_8/ReadVariableOp_1ReadVariableOp?model_3_model_2_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype028
6model_3/model_2/batch_normalization_8/ReadVariableOp_1?
Emodel_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpNmodel_3_model_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02G
Emodel_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
Gmodel_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPmodel_3_model_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02I
Gmodel_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
6model_3/model_2/batch_normalization_8/FusedBatchNormV3FusedBatchNormV30model_3/model_2/max_pooling2d_8/MaxPool:output:0<model_3/model_2/batch_normalization_8/ReadVariableOp:value:0>model_3/model_2/batch_normalization_8/ReadVariableOp_1:value:0Mmodel_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Omodel_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 28
6model_3/model_2/batch_normalization_8/FusedBatchNormV3?
"model_3/model_2/dropout_8/IdentityIdentity:model_3/model_2/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2$
"model_3/model_2/dropout_8/Identity?
.model_3/model_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp7model_3_model_2_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype020
.model_3/model_2/conv2d_9/Conv2D/ReadVariableOp?
model_3/model_2/conv2d_9/Conv2DConv2D+model_3/model_2/dropout_8/Identity:output:06model_3/model_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2!
model_3/model_2/conv2d_9/Conv2D?
/model_3/model_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp8model_3_model_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model_3/model_2/conv2d_9/BiasAdd/ReadVariableOp?
 model_3/model_2/conv2d_9/BiasAddBiasAdd(model_3/model_2/conv2d_9/Conv2D:output:07model_3/model_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2"
 model_3/model_2/conv2d_9/BiasAdd?
model_3/model_2/conv2d_9/ReluRelu)model_3/model_2/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?2
model_3/model_2/conv2d_9/Relu?
4model_3/model_2/batch_normalization_9/ReadVariableOpReadVariableOp=model_3_model_2_batch_normalization_9_readvariableop_resource*
_output_shapes	
:?*
dtype026
4model_3/model_2/batch_normalization_9/ReadVariableOp?
6model_3/model_2/batch_normalization_9/ReadVariableOp_1ReadVariableOp?model_3_model_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype028
6model_3/model_2/batch_normalization_9/ReadVariableOp_1?
Emodel_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpNmodel_3_model_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02G
Emodel_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
Gmodel_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPmodel_3_model_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02I
Gmodel_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
6model_3/model_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3+model_3/model_2/conv2d_9/Relu:activations:0<model_3/model_2/batch_normalization_9/ReadVariableOp:value:0>model_3/model_2/batch_normalization_9/ReadVariableOp_1:value:0Mmodel_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Omodel_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 28
6model_3/model_2/batch_normalization_9/FusedBatchNormV3?
'model_3/model_2/max_pooling2d_9/MaxPoolMaxPool:model_3/model_2/batch_normalization_9/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2)
'model_3/model_2/max_pooling2d_9/MaxPool?
"model_3/model_2/dropout_9/IdentityIdentity0model_3/model_2/max_pooling2d_9/MaxPool:output:0*
T0*0
_output_shapes
:??????????2$
"model_3/model_2/dropout_9/Identity?
Amodel_3/model_2/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_3/model_2/global_average_pooling2d_1/Mean/reduction_indices?
/model_3/model_2/global_average_pooling2d_1/MeanMean+model_3/model_2/dropout_9/Identity:output:0Jmodel_3/model_2/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????21
/model_3/model_2/global_average_pooling2d_1/Mean?
-model_3/model_2/dense_1/MatMul/ReadVariableOpReadVariableOp6model_3_model_2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-model_3/model_2/dense_1/MatMul/ReadVariableOp?
model_3/model_2/dense_1/MatMulMatMul8model_3/model_2/global_average_pooling2d_1/Mean:output:05model_3/model_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
model_3/model_2/dense_1/MatMul?
.model_3/model_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp7model_3_model_2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model_3/model_2/dense_1/BiasAdd/ReadVariableOp?
model_3/model_2/dense_1/BiasAddBiasAdd(model_3/model_2/dense_1/MatMul:product:06model_3/model_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
model_3/model_2/dense_1/BiasAdd?
0model_3/model_2/conv2d_6/Conv2D_1/ReadVariableOpReadVariableOp7model_3_model_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype022
0model_3/model_2/conv2d_6/Conv2D_1/ReadVariableOp?
!model_3/model_2/conv2d_6/Conv2D_1Conv2Dinput_58model_3/model_2/conv2d_6/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2#
!model_3/model_2/conv2d_6/Conv2D_1?
1model_3/model_2/conv2d_6/BiasAdd_1/ReadVariableOpReadVariableOp8model_3_model_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1model_3/model_2/conv2d_6/BiasAdd_1/ReadVariableOp?
"model_3/model_2/conv2d_6/BiasAdd_1BiasAdd*model_3/model_2/conv2d_6/Conv2D_1:output:09model_3/model_2/conv2d_6/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2$
"model_3/model_2/conv2d_6/BiasAdd_1?
model_3/model_2/conv2d_6/Relu_1Relu+model_3/model_2/conv2d_6/BiasAdd_1:output:0*
T0*1
_output_shapes
:???????????@2!
model_3/model_2/conv2d_6/Relu_1?
)model_3/model_2/max_pooling2d_6/MaxPool_1MaxPool-model_3/model_2/conv2d_6/Relu_1:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2+
)model_3/model_2/max_pooling2d_6/MaxPool_1?
6model_3/model_2/batch_normalization_6/ReadVariableOp_2ReadVariableOp=model_3_model_2_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype028
6model_3/model_2/batch_normalization_6/ReadVariableOp_2?
6model_3/model_2/batch_normalization_6/ReadVariableOp_3ReadVariableOp?model_3_model_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6model_3/model_2/batch_normalization_6/ReadVariableOp_3?
Gmodel_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpReadVariableOpNmodel_3_model_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gmodel_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp?
Imodel_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpPmodel_3_model_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02K
Imodel_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1?
8model_3/model_2/batch_normalization_6/FusedBatchNormV3_1FusedBatchNormV32model_3/model_2/max_pooling2d_6/MaxPool_1:output:0>model_3/model_2/batch_normalization_6/ReadVariableOp_2:value:0>model_3/model_2/batch_normalization_6/ReadVariableOp_3:value:0Omodel_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp:value:0Qmodel_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2:
8model_3/model_2/batch_normalization_6/FusedBatchNormV3_1?
$model_3/model_2/dropout_6/Identity_1Identity<model_3/model_2/batch_normalization_6/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:???????????@2&
$model_3/model_2/dropout_6/Identity_1?
0model_3/model_2/conv2d_7/Conv2D_1/ReadVariableOpReadVariableOp7model_3_model_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype022
0model_3/model_2/conv2d_7/Conv2D_1/ReadVariableOp?
!model_3/model_2/conv2d_7/Conv2D_1Conv2D-model_3/model_2/dropout_6/Identity_1:output:08model_3/model_2/conv2d_7/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2#
!model_3/model_2/conv2d_7/Conv2D_1?
1model_3/model_2/conv2d_7/BiasAdd_1/ReadVariableOpReadVariableOp8model_3_model_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model_3/model_2/conv2d_7/BiasAdd_1/ReadVariableOp?
"model_3/model_2/conv2d_7/BiasAdd_1BiasAdd*model_3/model_2/conv2d_7/Conv2D_1:output:09model_3/model_2/conv2d_7/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2$
"model_3/model_2/conv2d_7/BiasAdd_1?
model_3/model_2/conv2d_7/Relu_1Relu+model_3/model_2/conv2d_7/BiasAdd_1:output:0*
T0*2
_output_shapes 
:????????????2!
model_3/model_2/conv2d_7/Relu_1?
)model_3/model_2/max_pooling2d_7/MaxPool_1MaxPool-model_3/model_2/conv2d_7/Relu_1:activations:0*0
_output_shapes
:?????????@@?*
ksize
*
paddingVALID*
strides
2+
)model_3/model_2/max_pooling2d_7/MaxPool_1?
6model_3/model_2/batch_normalization_7/ReadVariableOp_2ReadVariableOp=model_3_model_2_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype028
6model_3/model_2/batch_normalization_7/ReadVariableOp_2?
6model_3/model_2/batch_normalization_7/ReadVariableOp_3ReadVariableOp?model_3_model_2_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype028
6model_3/model_2/batch_normalization_7/ReadVariableOp_3?
Gmodel_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpReadVariableOpNmodel_3_model_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02I
Gmodel_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp?
Imodel_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpPmodel_3_model_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02K
Imodel_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1?
8model_3/model_2/batch_normalization_7/FusedBatchNormV3_1FusedBatchNormV32model_3/model_2/max_pooling2d_7/MaxPool_1:output:0>model_3/model_2/batch_normalization_7/ReadVariableOp_2:value:0>model_3/model_2/batch_normalization_7/ReadVariableOp_3:value:0Omodel_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp:value:0Qmodel_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
is_training( 2:
8model_3/model_2/batch_normalization_7/FusedBatchNormV3_1?
$model_3/model_2/dropout_7/Identity_1Identity<model_3/model_2/batch_normalization_7/FusedBatchNormV3_1:y:0*
T0*0
_output_shapes
:?????????@@?2&
$model_3/model_2/dropout_7/Identity_1?
0model_3/model_2/conv2d_8/Conv2D_1/ReadVariableOpReadVariableOp7model_3_model_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0model_3/model_2/conv2d_8/Conv2D_1/ReadVariableOp?
!model_3/model_2/conv2d_8/Conv2D_1Conv2D-model_3/model_2/dropout_7/Identity_1:output:08model_3/model_2/conv2d_8/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2#
!model_3/model_2/conv2d_8/Conv2D_1?
1model_3/model_2/conv2d_8/BiasAdd_1/ReadVariableOpReadVariableOp8model_3_model_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model_3/model_2/conv2d_8/BiasAdd_1/ReadVariableOp?
"model_3/model_2/conv2d_8/BiasAdd_1BiasAdd*model_3/model_2/conv2d_8/Conv2D_1:output:09model_3/model_2/conv2d_8/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2$
"model_3/model_2/conv2d_8/BiasAdd_1?
model_3/model_2/conv2d_8/Relu_1Relu+model_3/model_2/conv2d_8/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????@@?2!
model_3/model_2/conv2d_8/Relu_1?
)model_3/model_2/max_pooling2d_8/MaxPool_1MaxPool-model_3/model_2/conv2d_8/Relu_1:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2+
)model_3/model_2/max_pooling2d_8/MaxPool_1?
6model_3/model_2/batch_normalization_8/ReadVariableOp_2ReadVariableOp=model_3_model_2_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype028
6model_3/model_2/batch_normalization_8/ReadVariableOp_2?
6model_3/model_2/batch_normalization_8/ReadVariableOp_3ReadVariableOp?model_3_model_2_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype028
6model_3/model_2/batch_normalization_8/ReadVariableOp_3?
Gmodel_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOpReadVariableOpNmodel_3_model_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02I
Gmodel_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp?
Imodel_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpPmodel_3_model_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02K
Imodel_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1?
8model_3/model_2/batch_normalization_8/FusedBatchNormV3_1FusedBatchNormV32model_3/model_2/max_pooling2d_8/MaxPool_1:output:0>model_3/model_2/batch_normalization_8/ReadVariableOp_2:value:0>model_3/model_2/batch_normalization_8/ReadVariableOp_3:value:0Omodel_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp:value:0Qmodel_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 2:
8model_3/model_2/batch_normalization_8/FusedBatchNormV3_1?
$model_3/model_2/dropout_8/Identity_1Identity<model_3/model_2/batch_normalization_8/FusedBatchNormV3_1:y:0*
T0*0
_output_shapes
:?????????  ?2&
$model_3/model_2/dropout_8/Identity_1?
0model_3/model_2/conv2d_9/Conv2D_1/ReadVariableOpReadVariableOp7model_3_model_2_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype022
0model_3/model_2/conv2d_9/Conv2D_1/ReadVariableOp?
!model_3/model_2/conv2d_9/Conv2D_1Conv2D-model_3/model_2/dropout_8/Identity_1:output:08model_3/model_2/conv2d_9/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2#
!model_3/model_2/conv2d_9/Conv2D_1?
1model_3/model_2/conv2d_9/BiasAdd_1/ReadVariableOpReadVariableOp8model_3_model_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype023
1model_3/model_2/conv2d_9/BiasAdd_1/ReadVariableOp?
"model_3/model_2/conv2d_9/BiasAdd_1BiasAdd*model_3/model_2/conv2d_9/Conv2D_1:output:09model_3/model_2/conv2d_9/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2$
"model_3/model_2/conv2d_9/BiasAdd_1?
model_3/model_2/conv2d_9/Relu_1Relu+model_3/model_2/conv2d_9/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????  ?2!
model_3/model_2/conv2d_9/Relu_1?
6model_3/model_2/batch_normalization_9/ReadVariableOp_2ReadVariableOp=model_3_model_2_batch_normalization_9_readvariableop_resource*
_output_shapes	
:?*
dtype028
6model_3/model_2/batch_normalization_9/ReadVariableOp_2?
6model_3/model_2/batch_normalization_9/ReadVariableOp_3ReadVariableOp?model_3_model_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype028
6model_3/model_2/batch_normalization_9/ReadVariableOp_3?
Gmodel_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOpReadVariableOpNmodel_3_model_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02I
Gmodel_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp?
Imodel_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpPmodel_3_model_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02K
Imodel_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1?
8model_3/model_2/batch_normalization_9/FusedBatchNormV3_1FusedBatchNormV3-model_3/model_2/conv2d_9/Relu_1:activations:0>model_3/model_2/batch_normalization_9/ReadVariableOp_2:value:0>model_3/model_2/batch_normalization_9/ReadVariableOp_3:value:0Omodel_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp:value:0Qmodel_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 2:
8model_3/model_2/batch_normalization_9/FusedBatchNormV3_1?
)model_3/model_2/max_pooling2d_9/MaxPool_1MaxPool<model_3/model_2/batch_normalization_9/FusedBatchNormV3_1:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2+
)model_3/model_2/max_pooling2d_9/MaxPool_1?
$model_3/model_2/dropout_9/Identity_1Identity2model_3/model_2/max_pooling2d_9/MaxPool_1:output:0*
T0*0
_output_shapes
:??????????2&
$model_3/model_2/dropout_9/Identity_1?
Cmodel_3/model_2/global_average_pooling2d_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2E
Cmodel_3/model_2/global_average_pooling2d_1/Mean_1/reduction_indices?
1model_3/model_2/global_average_pooling2d_1/Mean_1Mean-model_3/model_2/dropout_9/Identity_1:output:0Lmodel_3/model_2/global_average_pooling2d_1/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:??????????23
1model_3/model_2/global_average_pooling2d_1/Mean_1?
/model_3/model_2/dense_1/MatMul_1/ReadVariableOpReadVariableOp6model_3_model_2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/model_3/model_2/dense_1/MatMul_1/ReadVariableOp?
 model_3/model_2/dense_1/MatMul_1MatMul:model_3/model_2/global_average_pooling2d_1/Mean_1:output:07model_3/model_2/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 model_3/model_2/dense_1/MatMul_1?
0model_3/model_2/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp7model_3_model_2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0model_3/model_2/dense_1/BiasAdd_1/ReadVariableOp?
!model_3/model_2/dense_1/BiasAdd_1BiasAdd*model_3/model_2/dense_1/MatMul_1:product:08model_3/model_2/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!model_3/model_2/dense_1/BiasAdd_1?
model_3/lambda_1/subSub(model_3/model_2/dense_1/BiasAdd:output:0*model_3/model_2/dense_1/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
model_3/lambda_1/sub?
model_3/lambda_1/SquareSquaremodel_3/lambda_1/sub:z:0*
T0*(
_output_shapes
:??????????2
model_3/lambda_1/Square?
&model_3/lambda_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_3/lambda_1/Sum/reduction_indices?
model_3/lambda_1/SumSummodel_3/lambda_1/Square:y:0/model_3/lambda_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
model_3/lambda_1/Sum}
model_3/lambda_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
model_3/lambda_1/Maximum/y?
model_3/lambda_1/MaximumMaximummodel_3/lambda_1/Sum:output:0#model_3/lambda_1/Maximum/y:output:0*
T0*'
_output_shapes
:?????????2
model_3/lambda_1/Maximumu
model_3/lambda_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_3/lambda_1/Const?
model_3/lambda_1/Maximum_1Maximummodel_3/lambda_1/Maximum:z:0model_3/lambda_1/Const:output:0*
T0*'
_output_shapes
:?????????2
model_3/lambda_1/Maximum_1?
model_3/lambda_1/SqrtSqrtmodel_3/lambda_1/Maximum_1:z:0*
T0*'
_output_shapes
:?????????2
model_3/lambda_1/Sqrtt
IdentityIdentitymodel_3/lambda_1/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOpF^model_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpH^model_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1H^model_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpJ^model_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_15^model_3/model_2/batch_normalization_6/ReadVariableOp7^model_3/model_2/batch_normalization_6/ReadVariableOp_17^model_3/model_2/batch_normalization_6/ReadVariableOp_27^model_3/model_2/batch_normalization_6/ReadVariableOp_3F^model_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpH^model_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1H^model_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpJ^model_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_15^model_3/model_2/batch_normalization_7/ReadVariableOp7^model_3/model_2/batch_normalization_7/ReadVariableOp_17^model_3/model_2/batch_normalization_7/ReadVariableOp_27^model_3/model_2/batch_normalization_7/ReadVariableOp_3F^model_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOpH^model_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1H^model_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOpJ^model_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_15^model_3/model_2/batch_normalization_8/ReadVariableOp7^model_3/model_2/batch_normalization_8/ReadVariableOp_17^model_3/model_2/batch_normalization_8/ReadVariableOp_27^model_3/model_2/batch_normalization_8/ReadVariableOp_3F^model_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpH^model_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1H^model_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOpJ^model_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_15^model_3/model_2/batch_normalization_9/ReadVariableOp7^model_3/model_2/batch_normalization_9/ReadVariableOp_17^model_3/model_2/batch_normalization_9/ReadVariableOp_27^model_3/model_2/batch_normalization_9/ReadVariableOp_30^model_3/model_2/conv2d_6/BiasAdd/ReadVariableOp2^model_3/model_2/conv2d_6/BiasAdd_1/ReadVariableOp/^model_3/model_2/conv2d_6/Conv2D/ReadVariableOp1^model_3/model_2/conv2d_6/Conv2D_1/ReadVariableOp0^model_3/model_2/conv2d_7/BiasAdd/ReadVariableOp2^model_3/model_2/conv2d_7/BiasAdd_1/ReadVariableOp/^model_3/model_2/conv2d_7/Conv2D/ReadVariableOp1^model_3/model_2/conv2d_7/Conv2D_1/ReadVariableOp0^model_3/model_2/conv2d_8/BiasAdd/ReadVariableOp2^model_3/model_2/conv2d_8/BiasAdd_1/ReadVariableOp/^model_3/model_2/conv2d_8/Conv2D/ReadVariableOp1^model_3/model_2/conv2d_8/Conv2D_1/ReadVariableOp0^model_3/model_2/conv2d_9/BiasAdd/ReadVariableOp2^model_3/model_2/conv2d_9/BiasAdd_1/ReadVariableOp/^model_3/model_2/conv2d_9/Conv2D/ReadVariableOp1^model_3/model_2/conv2d_9/Conv2D_1/ReadVariableOp/^model_3/model_2/dense_1/BiasAdd/ReadVariableOp1^model_3/model_2/dense_1/BiasAdd_1/ReadVariableOp.^model_3/model_2/dense_1/MatMul/ReadVariableOp0^model_3/model_2/dense_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Emodel_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpEmodel_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
Gmodel_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Gmodel_3/model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12?
Gmodel_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpGmodel_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp2?
Imodel_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1Imodel_3/model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_12l
4model_3/model_2/batch_normalization_6/ReadVariableOp4model_3/model_2/batch_normalization_6/ReadVariableOp2p
6model_3/model_2/batch_normalization_6/ReadVariableOp_16model_3/model_2/batch_normalization_6/ReadVariableOp_12p
6model_3/model_2/batch_normalization_6/ReadVariableOp_26model_3/model_2/batch_normalization_6/ReadVariableOp_22p
6model_3/model_2/batch_normalization_6/ReadVariableOp_36model_3/model_2/batch_normalization_6/ReadVariableOp_32?
Emodel_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpEmodel_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
Gmodel_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Gmodel_3/model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12?
Gmodel_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpGmodel_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp2?
Imodel_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1Imodel_3/model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_12l
4model_3/model_2/batch_normalization_7/ReadVariableOp4model_3/model_2/batch_normalization_7/ReadVariableOp2p
6model_3/model_2/batch_normalization_7/ReadVariableOp_16model_3/model_2/batch_normalization_7/ReadVariableOp_12p
6model_3/model_2/batch_normalization_7/ReadVariableOp_26model_3/model_2/batch_normalization_7/ReadVariableOp_22p
6model_3/model_2/batch_normalization_7/ReadVariableOp_36model_3/model_2/batch_normalization_7/ReadVariableOp_32?
Emodel_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOpEmodel_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
Gmodel_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Gmodel_3/model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12?
Gmodel_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOpGmodel_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp2?
Imodel_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1Imodel_3/model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_12l
4model_3/model_2/batch_normalization_8/ReadVariableOp4model_3/model_2/batch_normalization_8/ReadVariableOp2p
6model_3/model_2/batch_normalization_8/ReadVariableOp_16model_3/model_2/batch_normalization_8/ReadVariableOp_12p
6model_3/model_2/batch_normalization_8/ReadVariableOp_26model_3/model_2/batch_normalization_8/ReadVariableOp_22p
6model_3/model_2/batch_normalization_8/ReadVariableOp_36model_3/model_2/batch_normalization_8/ReadVariableOp_32?
Emodel_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpEmodel_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
Gmodel_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Gmodel_3/model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12?
Gmodel_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOpGmodel_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp2?
Imodel_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1Imodel_3/model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_12l
4model_3/model_2/batch_normalization_9/ReadVariableOp4model_3/model_2/batch_normalization_9/ReadVariableOp2p
6model_3/model_2/batch_normalization_9/ReadVariableOp_16model_3/model_2/batch_normalization_9/ReadVariableOp_12p
6model_3/model_2/batch_normalization_9/ReadVariableOp_26model_3/model_2/batch_normalization_9/ReadVariableOp_22p
6model_3/model_2/batch_normalization_9/ReadVariableOp_36model_3/model_2/batch_normalization_9/ReadVariableOp_32b
/model_3/model_2/conv2d_6/BiasAdd/ReadVariableOp/model_3/model_2/conv2d_6/BiasAdd/ReadVariableOp2f
1model_3/model_2/conv2d_6/BiasAdd_1/ReadVariableOp1model_3/model_2/conv2d_6/BiasAdd_1/ReadVariableOp2`
.model_3/model_2/conv2d_6/Conv2D/ReadVariableOp.model_3/model_2/conv2d_6/Conv2D/ReadVariableOp2d
0model_3/model_2/conv2d_6/Conv2D_1/ReadVariableOp0model_3/model_2/conv2d_6/Conv2D_1/ReadVariableOp2b
/model_3/model_2/conv2d_7/BiasAdd/ReadVariableOp/model_3/model_2/conv2d_7/BiasAdd/ReadVariableOp2f
1model_3/model_2/conv2d_7/BiasAdd_1/ReadVariableOp1model_3/model_2/conv2d_7/BiasAdd_1/ReadVariableOp2`
.model_3/model_2/conv2d_7/Conv2D/ReadVariableOp.model_3/model_2/conv2d_7/Conv2D/ReadVariableOp2d
0model_3/model_2/conv2d_7/Conv2D_1/ReadVariableOp0model_3/model_2/conv2d_7/Conv2D_1/ReadVariableOp2b
/model_3/model_2/conv2d_8/BiasAdd/ReadVariableOp/model_3/model_2/conv2d_8/BiasAdd/ReadVariableOp2f
1model_3/model_2/conv2d_8/BiasAdd_1/ReadVariableOp1model_3/model_2/conv2d_8/BiasAdd_1/ReadVariableOp2`
.model_3/model_2/conv2d_8/Conv2D/ReadVariableOp.model_3/model_2/conv2d_8/Conv2D/ReadVariableOp2d
0model_3/model_2/conv2d_8/Conv2D_1/ReadVariableOp0model_3/model_2/conv2d_8/Conv2D_1/ReadVariableOp2b
/model_3/model_2/conv2d_9/BiasAdd/ReadVariableOp/model_3/model_2/conv2d_9/BiasAdd/ReadVariableOp2f
1model_3/model_2/conv2d_9/BiasAdd_1/ReadVariableOp1model_3/model_2/conv2d_9/BiasAdd_1/ReadVariableOp2`
.model_3/model_2/conv2d_9/Conv2D/ReadVariableOp.model_3/model_2/conv2d_9/Conv2D/ReadVariableOp2d
0model_3/model_2/conv2d_9/Conv2D_1/ReadVariableOp0model_3/model_2/conv2d_9/Conv2D_1/ReadVariableOp2`
.model_3/model_2/dense_1/BiasAdd/ReadVariableOp.model_3/model_2/dense_1/BiasAdd/ReadVariableOp2d
0model_3/model_2/dense_1/BiasAdd_1/ReadVariableOp0model_3/model_2/dense_1/BiasAdd_1/ReadVariableOp2^
-model_3/model_2/dense_1/MatMul/ReadVariableOp-model_3/model_2/dense_1/MatMul/ReadVariableOp2b
/model_3/model_2/dense_1/MatMul_1/ReadVariableOp/model_3/model_2/dense_1/MatMul_1/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_5
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69735

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_model_3_layer_call_and_return_conditional_losses_68120
input_4
input_5'
model_2_68038:@
model_2_68040:@
model_2_68042:@
model_2_68044:@
model_2_68046:@
model_2_68048:@(
model_2_68050:@?
model_2_68052:	?
model_2_68054:	?
model_2_68056:	?
model_2_68058:	?
model_2_68060:	?)
model_2_68062:??
model_2_68064:	?
model_2_68066:	?
model_2_68068:	?
model_2_68070:	?
model_2_68072:	?)
model_2_68074:??
model_2_68076:	?
model_2_68078:	?
model_2_68080:	?
model_2_68082:	?
model_2_68084:	?!
model_2_68086:
??
model_2_68088:	?
identity??model_2/StatefulPartitionedCall?!model_2/StatefulPartitionedCall_1?
model_2/StatefulPartitionedCallStatefulPartitionedCallinput_4model_2_68038model_2_68040model_2_68042model_2_68044model_2_68046model_2_68048model_2_68050model_2_68052model_2_68054model_2_68056model_2_68058model_2_68060model_2_68062model_2_68064model_2_68066model_2_68068model_2_68070model_2_68072model_2_68074model_2_68076model_2_68078model_2_68080model_2_68082model_2_68084model_2_68086model_2_68088*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_672402!
model_2/StatefulPartitionedCall?
!model_2/StatefulPartitionedCall_1StatefulPartitionedCallinput_5model_2_68038model_2_68040model_2_68042model_2_68044model_2_68046model_2_68048model_2_68050model_2_68052model_2_68054model_2_68056model_2_68058model_2_68060model_2_68062model_2_68064model_2_68066model_2_68068model_2_68070model_2_68072model_2_68074model_2_68076model_2_68078model_2_68080model_2_68082model_2_68084model_2_68086model_2_68088 ^model_2/StatefulPartitionedCall*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_672402#
!model_2/StatefulPartitionedCall_1?
lambda_1/PartitionedCallPartitionedCall(model_2/StatefulPartitionedCall:output:0*model_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_lambda_1_layer_call_and_return_conditional_losses_676852
lambda_1/PartitionedCall|
IdentityIdentity!lambda_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^model_2/StatefulPartitionedCall"^model_2/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall2F
!model_2/StatefulPartitionedCall_1!model_2/StatefulPartitionedCall_1:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_5
?
f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_66139

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_66174

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_9_layer_call_fn_69861

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_666762
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
b
D__inference_dropout_9_layer_call_and_return_conditional_losses_66683

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_7_layer_call_fn_69479

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_669992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
??
?-
!__inference__traced_restore_70377
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: <
"assignvariableop_5_conv2d_6_kernel:@.
 assignvariableop_6_conv2d_6_bias:@<
.assignvariableop_7_batch_normalization_6_gamma:@;
-assignvariableop_8_batch_normalization_6_beta:@B
4assignvariableop_9_batch_normalization_6_moving_mean:@G
9assignvariableop_10_batch_normalization_6_moving_variance:@>
#assignvariableop_11_conv2d_7_kernel:@?0
!assignvariableop_12_conv2d_7_bias:	?>
/assignvariableop_13_batch_normalization_7_gamma:	?=
.assignvariableop_14_batch_normalization_7_beta:	?D
5assignvariableop_15_batch_normalization_7_moving_mean:	?H
9assignvariableop_16_batch_normalization_7_moving_variance:	??
#assignvariableop_17_conv2d_8_kernel:??0
!assignvariableop_18_conv2d_8_bias:	?>
/assignvariableop_19_batch_normalization_8_gamma:	?=
.assignvariableop_20_batch_normalization_8_beta:	?D
5assignvariableop_21_batch_normalization_8_moving_mean:	?H
9assignvariableop_22_batch_normalization_8_moving_variance:	??
#assignvariableop_23_conv2d_9_kernel:??0
!assignvariableop_24_conv2d_9_bias:	?>
/assignvariableop_25_batch_normalization_9_gamma:	?=
.assignvariableop_26_batch_normalization_9_beta:	?D
5assignvariableop_27_batch_normalization_9_moving_mean:	?H
9assignvariableop_28_batch_normalization_9_moving_variance:	?6
"assignvariableop_29_dense_1_kernel:
??/
 assignvariableop_30_dense_1_bias:	?#
assignvariableop_31_total: #
assignvariableop_32_count: D
*assignvariableop_33_adam_conv2d_6_kernel_m:@6
(assignvariableop_34_adam_conv2d_6_bias_m:@D
6assignvariableop_35_adam_batch_normalization_6_gamma_m:@C
5assignvariableop_36_adam_batch_normalization_6_beta_m:@E
*assignvariableop_37_adam_conv2d_7_kernel_m:@?7
(assignvariableop_38_adam_conv2d_7_bias_m:	?E
6assignvariableop_39_adam_batch_normalization_7_gamma_m:	?D
5assignvariableop_40_adam_batch_normalization_7_beta_m:	?F
*assignvariableop_41_adam_conv2d_8_kernel_m:??7
(assignvariableop_42_adam_conv2d_8_bias_m:	?E
6assignvariableop_43_adam_batch_normalization_8_gamma_m:	?D
5assignvariableop_44_adam_batch_normalization_8_beta_m:	?F
*assignvariableop_45_adam_conv2d_9_kernel_m:??7
(assignvariableop_46_adam_conv2d_9_bias_m:	?E
6assignvariableop_47_adam_batch_normalization_9_gamma_m:	?D
5assignvariableop_48_adam_batch_normalization_9_beta_m:	?=
)assignvariableop_49_adam_dense_1_kernel_m:
??6
'assignvariableop_50_adam_dense_1_bias_m:	?D
*assignvariableop_51_adam_conv2d_6_kernel_v:@6
(assignvariableop_52_adam_conv2d_6_bias_v:@D
6assignvariableop_53_adam_batch_normalization_6_gamma_v:@C
5assignvariableop_54_adam_batch_normalization_6_beta_v:@E
*assignvariableop_55_adam_conv2d_7_kernel_v:@?7
(assignvariableop_56_adam_conv2d_7_bias_v:	?E
6assignvariableop_57_adam_batch_normalization_7_gamma_v:	?D
5assignvariableop_58_adam_batch_normalization_7_beta_v:	?F
*assignvariableop_59_adam_conv2d_8_kernel_v:??7
(assignvariableop_60_adam_conv2d_8_bias_v:	?E
6assignvariableop_61_adam_batch_normalization_8_gamma_v:	?D
5assignvariableop_62_adam_batch_normalization_8_beta_v:	?F
*assignvariableop_63_adam_conv2d_9_kernel_v:??7
(assignvariableop_64_adam_conv2d_9_bias_v:	?E
6assignvariableop_65_adam_batch_normalization_9_gamma_v:	?D
5assignvariableop_66_adam_batch_normalization_9_beta_v:	?=
)assignvariableop_67_adam_dense_1_kernel_v:
??6
'assignvariableop_68_adam_dense_1_bias_v:	?
identity_70??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*T
dtypesJ
H2F	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_6_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_6_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_6_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_6_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp4assignvariableop_9_batch_normalization_6_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp9assignvariableop_10_batch_normalization_6_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_7_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv2d_7_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_7_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp.assignvariableop_14_batch_normalization_7_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp5assignvariableop_15_batch_normalization_7_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp9assignvariableop_16_batch_normalization_7_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_conv2d_8_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp!assignvariableop_18_conv2d_8_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_8_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp.assignvariableop_20_batch_normalization_8_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp5assignvariableop_21_batch_normalization_8_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp9assignvariableop_22_batch_normalization_8_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp#assignvariableop_23_conv2d_9_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp!assignvariableop_24_conv2d_9_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp/assignvariableop_25_batch_normalization_9_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_batch_normalization_9_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp5assignvariableop_27_batch_normalization_9_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp9assignvariableop_28_batch_normalization_9_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp"assignvariableop_29_dense_1_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp assignvariableop_30_dense_1_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_totalIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_countIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_6_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_6_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_6_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_batch_normalization_6_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_7_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_7_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_7_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_7_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_8_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_8_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_batch_normalization_8_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_batch_normalization_8_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv2d_9_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv2d_9_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_batch_normalization_9_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_batch_normalization_9_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_1_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_1_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv2d_6_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv2d_6_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_6_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_6_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv2d_7_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv2d_7_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_7_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_7_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv2d_8_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv2d_8_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_8_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_batch_normalization_8_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv2d_9_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv2d_9_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_9_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_9_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp)assignvariableop_67_adam_dense_1_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp'assignvariableop_68_adam_dense_1_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_689
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_69Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_69f
Identity_70IdentityIdentity_69:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_70?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_70Identity_70:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
K
/__inference_max_pooling2d_9_layer_call_fn_69856

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_664132
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_66554

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_65878

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_65922

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_66218

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
m
C__inference_lambda_1_layer_call_and_return_conditional_losses_67603

inputs
inputs_1
identityV
subSubinputsinputs_1*
T0*(
_output_shapes
:??????????2
subV
SquareSquaresub:z:0*
T0*(
_output_shapes
:??????????2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Sum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
	Maximum/yq
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constp
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
	Maximum_1U
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_65843

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_9_layer_call_fn_69841

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_668402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_7_layer_call_fn_69453

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_660702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_69326

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_6_layer_call_fn_69262

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_659222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69391

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?%
B__inference_model_3_layer_call_and_return_conditional_losses_68374
inputs_0
inputs_1I
/model_2_conv2d_6_conv2d_readvariableop_resource:@>
0model_2_conv2d_6_biasadd_readvariableop_resource:@C
5model_2_batch_normalization_6_readvariableop_resource:@E
7model_2_batch_normalization_6_readvariableop_1_resource:@T
Fmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@V
Hmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@J
/model_2_conv2d_7_conv2d_readvariableop_resource:@??
0model_2_conv2d_7_biasadd_readvariableop_resource:	?D
5model_2_batch_normalization_7_readvariableop_resource:	?F
7model_2_batch_normalization_7_readvariableop_1_resource:	?U
Fmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	?W
Hmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	?K
/model_2_conv2d_8_conv2d_readvariableop_resource:???
0model_2_conv2d_8_biasadd_readvariableop_resource:	?D
5model_2_batch_normalization_8_readvariableop_resource:	?F
7model_2_batch_normalization_8_readvariableop_1_resource:	?U
Fmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	?W
Hmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	?K
/model_2_conv2d_9_conv2d_readvariableop_resource:???
0model_2_conv2d_9_biasadd_readvariableop_resource:	?D
5model_2_batch_normalization_9_readvariableop_resource:	?F
7model_2_batch_normalization_9_readvariableop_1_resource:	?U
Fmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	?W
Hmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	?B
.model_2_dense_1_matmul_readvariableop_resource:
??>
/model_2_dense_1_biasadd_readvariableop_resource:	?
identity??=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp??model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1??model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp?Amodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1?,model_2/batch_normalization_6/ReadVariableOp?.model_2/batch_normalization_6/ReadVariableOp_1?.model_2/batch_normalization_6/ReadVariableOp_2?.model_2/batch_normalization_6/ReadVariableOp_3?=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp??model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1??model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp?Amodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1?,model_2/batch_normalization_7/ReadVariableOp?.model_2/batch_normalization_7/ReadVariableOp_1?.model_2/batch_normalization_7/ReadVariableOp_2?.model_2/batch_normalization_7/ReadVariableOp_3?=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp??model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1??model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp?Amodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1?,model_2/batch_normalization_8/ReadVariableOp?.model_2/batch_normalization_8/ReadVariableOp_1?.model_2/batch_normalization_8/ReadVariableOp_2?.model_2/batch_normalization_8/ReadVariableOp_3?=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp??model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1??model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp?Amodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1?,model_2/batch_normalization_9/ReadVariableOp?.model_2/batch_normalization_9/ReadVariableOp_1?.model_2/batch_normalization_9/ReadVariableOp_2?.model_2/batch_normalization_9/ReadVariableOp_3?'model_2/conv2d_6/BiasAdd/ReadVariableOp?)model_2/conv2d_6/BiasAdd_1/ReadVariableOp?&model_2/conv2d_6/Conv2D/ReadVariableOp?(model_2/conv2d_6/Conv2D_1/ReadVariableOp?'model_2/conv2d_7/BiasAdd/ReadVariableOp?)model_2/conv2d_7/BiasAdd_1/ReadVariableOp?&model_2/conv2d_7/Conv2D/ReadVariableOp?(model_2/conv2d_7/Conv2D_1/ReadVariableOp?'model_2/conv2d_8/BiasAdd/ReadVariableOp?)model_2/conv2d_8/BiasAdd_1/ReadVariableOp?&model_2/conv2d_8/Conv2D/ReadVariableOp?(model_2/conv2d_8/Conv2D_1/ReadVariableOp?'model_2/conv2d_9/BiasAdd/ReadVariableOp?)model_2/conv2d_9/BiasAdd_1/ReadVariableOp?&model_2/conv2d_9/Conv2D/ReadVariableOp?(model_2/conv2d_9/Conv2D_1/ReadVariableOp?&model_2/dense_1/BiasAdd/ReadVariableOp?(model_2/dense_1/BiasAdd_1/ReadVariableOp?%model_2/dense_1/MatMul/ReadVariableOp?'model_2/dense_1/MatMul_1/ReadVariableOp?
&model_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&model_2/conv2d_6/Conv2D/ReadVariableOp?
model_2/conv2d_6/Conv2DConv2Dinputs_0.model_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
model_2/conv2d_6/Conv2D?
'model_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_2/conv2d_6/BiasAdd/ReadVariableOp?
model_2/conv2d_6/BiasAddBiasAdd model_2/conv2d_6/Conv2D:output:0/model_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
model_2/conv2d_6/BiasAdd?
model_2/conv2d_6/ReluRelu!model_2/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
model_2/conv2d_6/Relu?
model_2/max_pooling2d_6/MaxPoolMaxPool#model_2/conv2d_6/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2!
model_2/max_pooling2d_6/MaxPool?
,model_2/batch_normalization_6/ReadVariableOpReadVariableOp5model_2_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model_2/batch_normalization_6/ReadVariableOp?
.model_2/batch_normalization_6/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.model_2/batch_normalization_6/ReadVariableOp_1?
=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02?
=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02A
?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
.model_2/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3(model_2/max_pooling2d_6/MaxPool:output:04model_2/batch_normalization_6/ReadVariableOp:value:06model_2/batch_normalization_6/ReadVariableOp_1:value:0Emodel_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 20
.model_2/batch_normalization_6/FusedBatchNormV3?
model_2/dropout_6/IdentityIdentity2model_2/batch_normalization_6/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????@2
model_2/dropout_6/Identity?
&model_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02(
&model_2/conv2d_7/Conv2D/ReadVariableOp?
model_2/conv2d_7/Conv2DConv2D#model_2/dropout_6/Identity:output:0.model_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model_2/conv2d_7/Conv2D?
'model_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_2/conv2d_7/BiasAdd/ReadVariableOp?
model_2/conv2d_7/BiasAddBiasAdd model_2/conv2d_7/Conv2D:output:0/model_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
model_2/conv2d_7/BiasAdd?
model_2/conv2d_7/ReluRelu!model_2/conv2d_7/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
model_2/conv2d_7/Relu?
model_2/max_pooling2d_7/MaxPoolMaxPool#model_2/conv2d_7/Relu:activations:0*0
_output_shapes
:?????????@@?*
ksize
*
paddingVALID*
strides
2!
model_2/max_pooling2d_7/MaxPool?
,model_2/batch_normalization_7/ReadVariableOpReadVariableOp5model_2_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,model_2/batch_normalization_7/ReadVariableOp?
.model_2/batch_normalization_7/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_7/ReadVariableOp_1?
=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02A
?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
.model_2/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3(model_2/max_pooling2d_7/MaxPool:output:04model_2/batch_normalization_7/ReadVariableOp:value:06model_2/batch_normalization_7/ReadVariableOp_1:value:0Emodel_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
is_training( 20
.model_2/batch_normalization_7/FusedBatchNormV3?
model_2/dropout_7/IdentityIdentity2model_2/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@@?2
model_2/dropout_7/Identity?
&model_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&model_2/conv2d_8/Conv2D/ReadVariableOp?
model_2/conv2d_8/Conv2DConv2D#model_2/dropout_7/Identity:output:0.model_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
model_2/conv2d_8/Conv2D?
'model_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_2/conv2d_8/BiasAdd/ReadVariableOp?
model_2/conv2d_8/BiasAddBiasAdd model_2/conv2d_8/Conv2D:output:0/model_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2
model_2/conv2d_8/BiasAdd?
model_2/conv2d_8/ReluRelu!model_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
model_2/conv2d_8/Relu?
model_2/max_pooling2d_8/MaxPoolMaxPool#model_2/conv2d_8/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2!
model_2/max_pooling2d_8/MaxPool?
,model_2/batch_normalization_8/ReadVariableOpReadVariableOp5model_2_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,model_2/batch_normalization_8/ReadVariableOp?
.model_2/batch_normalization_8/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_8/ReadVariableOp_1?
=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02A
?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
.model_2/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3(model_2/max_pooling2d_8/MaxPool:output:04model_2/batch_normalization_8/ReadVariableOp:value:06model_2/batch_normalization_8/ReadVariableOp_1:value:0Emodel_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 20
.model_2/batch_normalization_8/FusedBatchNormV3?
model_2/dropout_8/IdentityIdentity2model_2/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
model_2/dropout_8/Identity?
&model_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp/model_2_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&model_2/conv2d_9/Conv2D/ReadVariableOp?
model_2/conv2d_9/Conv2DConv2D#model_2/dropout_8/Identity:output:0.model_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
model_2/conv2d_9/Conv2D?
'model_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_2/conv2d_9/BiasAdd/ReadVariableOp?
model_2/conv2d_9/BiasAddBiasAdd model_2/conv2d_9/Conv2D:output:0/model_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2
model_2/conv2d_9/BiasAdd?
model_2/conv2d_9/ReluRelu!model_2/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?2
model_2/conv2d_9/Relu?
,model_2/batch_normalization_9/ReadVariableOpReadVariableOp5model_2_batch_normalization_9_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,model_2/batch_normalization_9/ReadVariableOp?
.model_2/batch_normalization_9/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_9/ReadVariableOp_1?
=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02A
?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
.model_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3#model_2/conv2d_9/Relu:activations:04model_2/batch_normalization_9/ReadVariableOp:value:06model_2/batch_normalization_9/ReadVariableOp_1:value:0Emodel_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 20
.model_2/batch_normalization_9/FusedBatchNormV3?
model_2/max_pooling2d_9/MaxPoolMaxPool2model_2/batch_normalization_9/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2!
model_2/max_pooling2d_9/MaxPool?
model_2/dropout_9/IdentityIdentity(model_2/max_pooling2d_9/MaxPool:output:0*
T0*0
_output_shapes
:??????????2
model_2/dropout_9/Identity?
9model_2/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9model_2/global_average_pooling2d_1/Mean/reduction_indices?
'model_2/global_average_pooling2d_1/MeanMean#model_2/dropout_9/Identity:output:0Bmodel_2/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2)
'model_2/global_average_pooling2d_1/Mean?
%model_2/dense_1/MatMul/ReadVariableOpReadVariableOp.model_2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%model_2/dense_1/MatMul/ReadVariableOp?
model_2/dense_1/MatMulMatMul0model_2/global_average_pooling2d_1/Mean:output:0-model_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense_1/MatMul?
&model_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&model_2/dense_1/BiasAdd/ReadVariableOp?
model_2/dense_1/BiasAddBiasAdd model_2/dense_1/MatMul:product:0.model_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense_1/BiasAdd?
(model_2/conv2d_6/Conv2D_1/ReadVariableOpReadVariableOp/model_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(model_2/conv2d_6/Conv2D_1/ReadVariableOp?
model_2/conv2d_6/Conv2D_1Conv2Dinputs_10model_2/conv2d_6/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
model_2/conv2d_6/Conv2D_1?
)model_2/conv2d_6/BiasAdd_1/ReadVariableOpReadVariableOp0model_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_2/conv2d_6/BiasAdd_1/ReadVariableOp?
model_2/conv2d_6/BiasAdd_1BiasAdd"model_2/conv2d_6/Conv2D_1:output:01model_2/conv2d_6/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
model_2/conv2d_6/BiasAdd_1?
model_2/conv2d_6/Relu_1Relu#model_2/conv2d_6/BiasAdd_1:output:0*
T0*1
_output_shapes
:???????????@2
model_2/conv2d_6/Relu_1?
!model_2/max_pooling2d_6/MaxPool_1MaxPool%model_2/conv2d_6/Relu_1:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2#
!model_2/max_pooling2d_6/MaxPool_1?
.model_2/batch_normalization_6/ReadVariableOp_2ReadVariableOp5model_2_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype020
.model_2/batch_normalization_6/ReadVariableOp_2?
.model_2/batch_normalization_6/ReadVariableOp_3ReadVariableOp7model_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype020
.model_2/batch_normalization_6/ReadVariableOp_3?
?model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02A
?model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp?
Amodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02C
Amodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1?
0model_2/batch_normalization_6/FusedBatchNormV3_1FusedBatchNormV3*model_2/max_pooling2d_6/MaxPool_1:output:06model_2/batch_normalization_6/ReadVariableOp_2:value:06model_2/batch_normalization_6/ReadVariableOp_3:value:0Gmodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp:value:0Imodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 22
0model_2/batch_normalization_6/FusedBatchNormV3_1?
model_2/dropout_6/Identity_1Identity4model_2/batch_normalization_6/FusedBatchNormV3_1:y:0*
T0*1
_output_shapes
:???????????@2
model_2/dropout_6/Identity_1?
(model_2/conv2d_7/Conv2D_1/ReadVariableOpReadVariableOp/model_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02*
(model_2/conv2d_7/Conv2D_1/ReadVariableOp?
model_2/conv2d_7/Conv2D_1Conv2D%model_2/dropout_6/Identity_1:output:00model_2/conv2d_7/Conv2D_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
model_2/conv2d_7/Conv2D_1?
)model_2/conv2d_7/BiasAdd_1/ReadVariableOpReadVariableOp0model_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_2/conv2d_7/BiasAdd_1/ReadVariableOp?
model_2/conv2d_7/BiasAdd_1BiasAdd"model_2/conv2d_7/Conv2D_1:output:01model_2/conv2d_7/BiasAdd_1/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
model_2/conv2d_7/BiasAdd_1?
model_2/conv2d_7/Relu_1Relu#model_2/conv2d_7/BiasAdd_1:output:0*
T0*2
_output_shapes 
:????????????2
model_2/conv2d_7/Relu_1?
!model_2/max_pooling2d_7/MaxPool_1MaxPool%model_2/conv2d_7/Relu_1:activations:0*0
_output_shapes
:?????????@@?*
ksize
*
paddingVALID*
strides
2#
!model_2/max_pooling2d_7/MaxPool_1?
.model_2/batch_normalization_7/ReadVariableOp_2ReadVariableOp5model_2_batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_7/ReadVariableOp_2?
.model_2/batch_normalization_7/ReadVariableOp_3ReadVariableOp7model_2_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_7/ReadVariableOp_3?
?model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp?
Amodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02C
Amodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1?
0model_2/batch_normalization_7/FusedBatchNormV3_1FusedBatchNormV3*model_2/max_pooling2d_7/MaxPool_1:output:06model_2/batch_normalization_7/ReadVariableOp_2:value:06model_2/batch_normalization_7/ReadVariableOp_3:value:0Gmodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp:value:0Imodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
is_training( 22
0model_2/batch_normalization_7/FusedBatchNormV3_1?
model_2/dropout_7/Identity_1Identity4model_2/batch_normalization_7/FusedBatchNormV3_1:y:0*
T0*0
_output_shapes
:?????????@@?2
model_2/dropout_7/Identity_1?
(model_2/conv2d_8/Conv2D_1/ReadVariableOpReadVariableOp/model_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model_2/conv2d_8/Conv2D_1/ReadVariableOp?
model_2/conv2d_8/Conv2D_1Conv2D%model_2/dropout_7/Identity_1:output:00model_2/conv2d_8/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
model_2/conv2d_8/Conv2D_1?
)model_2/conv2d_8/BiasAdd_1/ReadVariableOpReadVariableOp0model_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_2/conv2d_8/BiasAdd_1/ReadVariableOp?
model_2/conv2d_8/BiasAdd_1BiasAdd"model_2/conv2d_8/Conv2D_1:output:01model_2/conv2d_8/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2
model_2/conv2d_8/BiasAdd_1?
model_2/conv2d_8/Relu_1Relu#model_2/conv2d_8/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????@@?2
model_2/conv2d_8/Relu_1?
!model_2/max_pooling2d_8/MaxPool_1MaxPool%model_2/conv2d_8/Relu_1:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2#
!model_2/max_pooling2d_8/MaxPool_1?
.model_2/batch_normalization_8/ReadVariableOp_2ReadVariableOp5model_2_batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_8/ReadVariableOp_2?
.model_2/batch_normalization_8/ReadVariableOp_3ReadVariableOp7model_2_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_8/ReadVariableOp_3?
?model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp?
Amodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02C
Amodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1?
0model_2/batch_normalization_8/FusedBatchNormV3_1FusedBatchNormV3*model_2/max_pooling2d_8/MaxPool_1:output:06model_2/batch_normalization_8/ReadVariableOp_2:value:06model_2/batch_normalization_8/ReadVariableOp_3:value:0Gmodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp:value:0Imodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 22
0model_2/batch_normalization_8/FusedBatchNormV3_1?
model_2/dropout_8/Identity_1Identity4model_2/batch_normalization_8/FusedBatchNormV3_1:y:0*
T0*0
_output_shapes
:?????????  ?2
model_2/dropout_8/Identity_1?
(model_2/conv2d_9/Conv2D_1/ReadVariableOpReadVariableOp/model_2_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(model_2/conv2d_9/Conv2D_1/ReadVariableOp?
model_2/conv2d_9/Conv2D_1Conv2D%model_2/dropout_8/Identity_1:output:00model_2/conv2d_9/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
model_2/conv2d_9/Conv2D_1?
)model_2/conv2d_9/BiasAdd_1/ReadVariableOpReadVariableOp0model_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_2/conv2d_9/BiasAdd_1/ReadVariableOp?
model_2/conv2d_9/BiasAdd_1BiasAdd"model_2/conv2d_9/Conv2D_1:output:01model_2/conv2d_9/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2
model_2/conv2d_9/BiasAdd_1?
model_2/conv2d_9/Relu_1Relu#model_2/conv2d_9/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????  ?2
model_2/conv2d_9/Relu_1?
.model_2/batch_normalization_9/ReadVariableOp_2ReadVariableOp5model_2_batch_normalization_9_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_9/ReadVariableOp_2?
.model_2/batch_normalization_9/ReadVariableOp_3ReadVariableOp7model_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_2/batch_normalization_9/ReadVariableOp_3?
?model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp?
Amodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02C
Amodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1?
0model_2/batch_normalization_9/FusedBatchNormV3_1FusedBatchNormV3%model_2/conv2d_9/Relu_1:activations:06model_2/batch_normalization_9/ReadVariableOp_2:value:06model_2/batch_normalization_9/ReadVariableOp_3:value:0Gmodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp:value:0Imodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 22
0model_2/batch_normalization_9/FusedBatchNormV3_1?
!model_2/max_pooling2d_9/MaxPool_1MaxPool4model_2/batch_normalization_9/FusedBatchNormV3_1:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2#
!model_2/max_pooling2d_9/MaxPool_1?
model_2/dropout_9/Identity_1Identity*model_2/max_pooling2d_9/MaxPool_1:output:0*
T0*0
_output_shapes
:??????????2
model_2/dropout_9/Identity_1?
;model_2/global_average_pooling2d_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;model_2/global_average_pooling2d_1/Mean_1/reduction_indices?
)model_2/global_average_pooling2d_1/Mean_1Mean%model_2/dropout_9/Identity_1:output:0Dmodel_2/global_average_pooling2d_1/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2+
)model_2/global_average_pooling2d_1/Mean_1?
'model_2/dense_1/MatMul_1/ReadVariableOpReadVariableOp.model_2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'model_2/dense_1/MatMul_1/ReadVariableOp?
model_2/dense_1/MatMul_1MatMul2model_2/global_average_pooling2d_1/Mean_1:output:0/model_2/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense_1/MatMul_1?
(model_2/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp/model_2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_2/dense_1/BiasAdd_1/ReadVariableOp?
model_2/dense_1/BiasAdd_1BiasAdd"model_2/dense_1/MatMul_1:product:00model_2/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_2/dense_1/BiasAdd_1?
lambda_1/subSub model_2/dense_1/BiasAdd:output:0"model_2/dense_1/BiasAdd_1:output:0*
T0*(
_output_shapes
:??????????2
lambda_1/subq
lambda_1/SquareSquarelambda_1/sub:z:0*
T0*(
_output_shapes
:??????????2
lambda_1/Square?
lambda_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2 
lambda_1/Sum/reduction_indices?
lambda_1/SumSumlambda_1/Square:y:0'lambda_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
lambda_1/Summ
lambda_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
lambda_1/Maximum/y?
lambda_1/MaximumMaximumlambda_1/Sum:output:0lambda_1/Maximum/y:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/Maximume
lambda_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda_1/Const?
lambda_1/Maximum_1Maximumlambda_1/Maximum:z:0lambda_1/Const:output:0*
T0*'
_output_shapes
:?????????2
lambda_1/Maximum_1p
lambda_1/SqrtSqrtlambda_1/Maximum_1:z:0*
T0*'
_output_shapes
:?????????2
lambda_1/Sqrtl
IdentityIdentitylambda_1/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp>^model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1@^model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOpB^model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1-^model_2/batch_normalization_6/ReadVariableOp/^model_2/batch_normalization_6/ReadVariableOp_1/^model_2/batch_normalization_6/ReadVariableOp_2/^model_2/batch_normalization_6/ReadVariableOp_3>^model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1@^model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOpB^model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1-^model_2/batch_normalization_7/ReadVariableOp/^model_2/batch_normalization_7/ReadVariableOp_1/^model_2/batch_normalization_7/ReadVariableOp_2/^model_2/batch_normalization_7/ReadVariableOp_3>^model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1@^model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOpB^model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1-^model_2/batch_normalization_8/ReadVariableOp/^model_2/batch_normalization_8/ReadVariableOp_1/^model_2/batch_normalization_8/ReadVariableOp_2/^model_2/batch_normalization_8/ReadVariableOp_3>^model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1@^model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOpB^model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1-^model_2/batch_normalization_9/ReadVariableOp/^model_2/batch_normalization_9/ReadVariableOp_1/^model_2/batch_normalization_9/ReadVariableOp_2/^model_2/batch_normalization_9/ReadVariableOp_3(^model_2/conv2d_6/BiasAdd/ReadVariableOp*^model_2/conv2d_6/BiasAdd_1/ReadVariableOp'^model_2/conv2d_6/Conv2D/ReadVariableOp)^model_2/conv2d_6/Conv2D_1/ReadVariableOp(^model_2/conv2d_7/BiasAdd/ReadVariableOp*^model_2/conv2d_7/BiasAdd_1/ReadVariableOp'^model_2/conv2d_7/Conv2D/ReadVariableOp)^model_2/conv2d_7/Conv2D_1/ReadVariableOp(^model_2/conv2d_8/BiasAdd/ReadVariableOp*^model_2/conv2d_8/BiasAdd_1/ReadVariableOp'^model_2/conv2d_8/Conv2D/ReadVariableOp)^model_2/conv2d_8/Conv2D_1/ReadVariableOp(^model_2/conv2d_9/BiasAdd/ReadVariableOp*^model_2/conv2d_9/BiasAdd_1/ReadVariableOp'^model_2/conv2d_9/Conv2D/ReadVariableOp)^model_2/conv2d_9/Conv2D_1/ReadVariableOp'^model_2/dense_1/BiasAdd/ReadVariableOp)^model_2/dense_1/BiasAdd_1/ReadVariableOp&^model_2/dense_1/MatMul/ReadVariableOp(^model_2/dense_1/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2~
=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2?
?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12?
?model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp?model_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp2?
Amodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_1Amodel_2/batch_normalization_6/FusedBatchNormV3_1/ReadVariableOp_12\
,model_2/batch_normalization_6/ReadVariableOp,model_2/batch_normalization_6/ReadVariableOp2`
.model_2/batch_normalization_6/ReadVariableOp_1.model_2/batch_normalization_6/ReadVariableOp_12`
.model_2/batch_normalization_6/ReadVariableOp_2.model_2/batch_normalization_6/ReadVariableOp_22`
.model_2/batch_normalization_6/ReadVariableOp_3.model_2/batch_normalization_6/ReadVariableOp_32~
=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2?
?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12?
?model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp?model_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp2?
Amodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_1Amodel_2/batch_normalization_7/FusedBatchNormV3_1/ReadVariableOp_12\
,model_2/batch_normalization_7/ReadVariableOp,model_2/batch_normalization_7/ReadVariableOp2`
.model_2/batch_normalization_7/ReadVariableOp_1.model_2/batch_normalization_7/ReadVariableOp_12`
.model_2/batch_normalization_7/ReadVariableOp_2.model_2/batch_normalization_7/ReadVariableOp_22`
.model_2/batch_normalization_7/ReadVariableOp_3.model_2/batch_normalization_7/ReadVariableOp_32~
=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2?
?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12?
?model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp?model_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp2?
Amodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_1Amodel_2/batch_normalization_8/FusedBatchNormV3_1/ReadVariableOp_12\
,model_2/batch_normalization_8/ReadVariableOp,model_2/batch_normalization_8/ReadVariableOp2`
.model_2/batch_normalization_8/ReadVariableOp_1.model_2/batch_normalization_8/ReadVariableOp_12`
.model_2/batch_normalization_8/ReadVariableOp_2.model_2/batch_normalization_8/ReadVariableOp_22`
.model_2/batch_normalization_8/ReadVariableOp_3.model_2/batch_normalization_8/ReadVariableOp_32~
=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12?
?model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp?model_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp2?
Amodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_1Amodel_2/batch_normalization_9/FusedBatchNormV3_1/ReadVariableOp_12\
,model_2/batch_normalization_9/ReadVariableOp,model_2/batch_normalization_9/ReadVariableOp2`
.model_2/batch_normalization_9/ReadVariableOp_1.model_2/batch_normalization_9/ReadVariableOp_12`
.model_2/batch_normalization_9/ReadVariableOp_2.model_2/batch_normalization_9/ReadVariableOp_22`
.model_2/batch_normalization_9/ReadVariableOp_3.model_2/batch_normalization_9/ReadVariableOp_32R
'model_2/conv2d_6/BiasAdd/ReadVariableOp'model_2/conv2d_6/BiasAdd/ReadVariableOp2V
)model_2/conv2d_6/BiasAdd_1/ReadVariableOp)model_2/conv2d_6/BiasAdd_1/ReadVariableOp2P
&model_2/conv2d_6/Conv2D/ReadVariableOp&model_2/conv2d_6/Conv2D/ReadVariableOp2T
(model_2/conv2d_6/Conv2D_1/ReadVariableOp(model_2/conv2d_6/Conv2D_1/ReadVariableOp2R
'model_2/conv2d_7/BiasAdd/ReadVariableOp'model_2/conv2d_7/BiasAdd/ReadVariableOp2V
)model_2/conv2d_7/BiasAdd_1/ReadVariableOp)model_2/conv2d_7/BiasAdd_1/ReadVariableOp2P
&model_2/conv2d_7/Conv2D/ReadVariableOp&model_2/conv2d_7/Conv2D/ReadVariableOp2T
(model_2/conv2d_7/Conv2D_1/ReadVariableOp(model_2/conv2d_7/Conv2D_1/ReadVariableOp2R
'model_2/conv2d_8/BiasAdd/ReadVariableOp'model_2/conv2d_8/BiasAdd/ReadVariableOp2V
)model_2/conv2d_8/BiasAdd_1/ReadVariableOp)model_2/conv2d_8/BiasAdd_1/ReadVariableOp2P
&model_2/conv2d_8/Conv2D/ReadVariableOp&model_2/conv2d_8/Conv2D/ReadVariableOp2T
(model_2/conv2d_8/Conv2D_1/ReadVariableOp(model_2/conv2d_8/Conv2D_1/ReadVariableOp2R
'model_2/conv2d_9/BiasAdd/ReadVariableOp'model_2/conv2d_9/BiasAdd/ReadVariableOp2V
)model_2/conv2d_9/BiasAdd_1/ReadVariableOp)model_2/conv2d_9/BiasAdd_1/ReadVariableOp2P
&model_2/conv2d_9/Conv2D/ReadVariableOp&model_2/conv2d_9/Conv2D/ReadVariableOp2T
(model_2/conv2d_9/Conv2D_1/ReadVariableOp(model_2/conv2d_9/Conv2D_1/ReadVariableOp2P
&model_2/dense_1/BiasAdd/ReadVariableOp&model_2/dense_1/BiasAdd/ReadVariableOp2T
(model_2/dense_1/BiasAdd_1/ReadVariableOp(model_2/dense_1/BiasAdd_1/ReadVariableOp2N
%model_2/dense_1/MatMul/ReadVariableOp%model_2/dense_1/MatMul/ReadVariableOp2R
'model_2/dense_1/MatMul_1/ReadVariableOp'model_2/dense_1/MatMul_1/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_69496

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????@@?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????@@?*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????@@?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????@@?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????@@?2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_7_layer_call_fn_69466

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_665542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_66497

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_66639

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_68734
inputs_0
inputs_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:
??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_678352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_66535

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????@@?*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_67948
input_4
input_5!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:
??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4input_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_678352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_5
?
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_69484

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????@@?2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????@@?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_66436

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_8_layer_call_fn_69692

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_666262
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_69920

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69789

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_69345

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????@@?*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
B__inference_model_3_layer_call_and_return_conditional_losses_68034
input_4
input_5'
model_2_67952:@
model_2_67954:@
model_2_67956:@
model_2_67958:@
model_2_67960:@
model_2_67962:@(
model_2_67964:@?
model_2_67966:	?
model_2_67968:	?
model_2_67970:	?
model_2_67972:	?
model_2_67974:	?)
model_2_67976:??
model_2_67978:	?
model_2_67980:	?
model_2_67982:	?
model_2_67984:	?
model_2_67986:	?)
model_2_67988:??
model_2_67990:	?
model_2_67992:	?
model_2_67994:	?
model_2_67996:	?
model_2_67998:	?!
model_2_68000:
??
model_2_68002:	?
identity??model_2/StatefulPartitionedCall?!model_2/StatefulPartitionedCall_1?
model_2/StatefulPartitionedCallStatefulPartitionedCallinput_4model_2_67952model_2_67954model_2_67956model_2_67958model_2_67960model_2_67962model_2_67964model_2_67966model_2_67968model_2_67970model_2_67972model_2_67974model_2_67976model_2_67978model_2_67980model_2_67982model_2_67984model_2_67986model_2_67988model_2_67990model_2_67992model_2_67994model_2_67996model_2_67998model_2_68000model_2_68002*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_667092!
model_2/StatefulPartitionedCall?
!model_2/StatefulPartitionedCall_1StatefulPartitionedCallinput_5model_2_67952model_2_67954model_2_67956model_2_67958model_2_67960model_2_67962model_2_67964model_2_67966model_2_67968model_2_67970model_2_67972model_2_67974model_2_67976model_2_67978model_2_67980model_2_67982model_2_67984model_2_67986model_2_67988model_2_67990model_2_67992model_2_67994model_2_67996model_2_67998model_2_68000model_2_68002*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_667092#
!model_2/StatefulPartitionedCall_1?
lambda_1/PartitionedCallPartitionedCall(model_2/StatefulPartitionedCall:output:0*model_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_lambda_1_layer_call_and_return_conditional_losses_676032
lambda_1/PartitionedCall|
IdentityIdentity!lambda_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^model_2/StatefulPartitionedCall"^model_2/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall2F
!model_2/StatefulPartitionedCall_1!model_2/StatefulPartitionedCall_1:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_5
?
c
D__inference_dropout_9_layer_call_and_return_conditional_losses_66799

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69427

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69600

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?X
?
B__inference_model_2_layer_call_and_return_conditional_losses_66709

inputs(
conv2d_6_66469:@
conv2d_6_66471:@)
batch_normalization_6_66498:@)
batch_normalization_6_66500:@)
batch_normalization_6_66502:@)
batch_normalization_6_66504:@)
conv2d_7_66526:@?
conv2d_7_66528:	?*
batch_normalization_7_66555:	?*
batch_normalization_7_66557:	?*
batch_normalization_7_66559:	?*
batch_normalization_7_66561:	?*
conv2d_8_66583:??
conv2d_8_66585:	?*
batch_normalization_8_66612:	?*
batch_normalization_8_66614:	?*
batch_normalization_8_66616:	?*
batch_normalization_8_66618:	?*
conv2d_9_66640:??
conv2d_9_66642:	?*
batch_normalization_9_66663:	?*
batch_normalization_9_66665:	?*
batch_normalization_9_66667:	?*
batch_normalization_9_66669:	?!
dense_1_66703:
??
dense_1_66705:	?
identity??-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_66469conv2d_6_66471*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_664682"
 conv2d_6/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_664782!
max_pooling2d_6/PartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0batch_normalization_6_66498batch_normalization_6_66500batch_normalization_6_66502batch_normalization_6_66504*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_664972/
-batch_normalization_6/StatefulPartitionedCall?
dropout_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_665122
dropout_6/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0conv2d_7_66526conv2d_7_66528*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_665252"
 conv2d_7/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_665352!
max_pooling2d_7/PartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0batch_normalization_7_66555batch_normalization_7_66557batch_normalization_7_66559batch_normalization_7_66561*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_665542/
-batch_normalization_7/StatefulPartitionedCall?
dropout_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_665692
dropout_7/PartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0conv2d_8_66583conv2d_8_66585*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_665822"
 conv2d_8/StatefulPartitionedCall?
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_665922!
max_pooling2d_8/PartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0batch_normalization_8_66612batch_normalization_8_66614batch_normalization_8_66616batch_normalization_8_66618*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_666112/
-batch_normalization_8/StatefulPartitionedCall?
dropout_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_666262
dropout_8/PartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv2d_9_66640conv2d_9_66642*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_666392"
 conv2d_9/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_9_66663batch_normalization_9_66665batch_normalization_9_66667batch_normalization_9_66669*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_666622/
-batch_normalization_9/StatefulPartitionedCall?
max_pooling2d_9/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_666762!
max_pooling2d_9/PartitionedCall?
dropout_9/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_666832
dropout_9/PartitionedCall?
*global_average_pooling2d_1/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *^
fYRW
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_666902,
*global_average_pooling2d_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0dense_1_66703dense_1_66705*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_667022!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_69894

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_66344

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69771

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_66592

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_9_layer_call_fn_69828

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_666622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_69900

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_66917

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69218

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_66690

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?X
?
B__inference_model_2_layer_call_and_return_conditional_losses_67426
input_6(
conv2d_6_67355:@
conv2d_6_67357:@)
batch_normalization_6_67361:@)
batch_normalization_6_67363:@)
batch_normalization_6_67365:@)
batch_normalization_6_67367:@)
conv2d_7_67371:@?
conv2d_7_67373:	?*
batch_normalization_7_67377:	?*
batch_normalization_7_67379:	?*
batch_normalization_7_67381:	?*
batch_normalization_7_67383:	?*
conv2d_8_67387:??
conv2d_8_67389:	?*
batch_normalization_8_67393:	?*
batch_normalization_8_67395:	?*
batch_normalization_8_67397:	?*
batch_normalization_8_67399:	?*
conv2d_9_67403:??
conv2d_9_67405:	?*
batch_normalization_9_67408:	?*
batch_normalization_9_67410:	?*
batch_normalization_9_67412:	?*
batch_normalization_9_67414:	?!
dense_1_67420:
??
dense_1_67422:	?
identity??-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_6conv2d_6_67355conv2d_6_67357*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_664682"
 conv2d_6/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_664782!
max_pooling2d_6/PartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0batch_normalization_6_67361batch_normalization_6_67363batch_normalization_6_67365batch_normalization_6_67367*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_664972/
-batch_normalization_6/StatefulPartitionedCall?
dropout_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_665122
dropout_6/PartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0conv2d_7_67371conv2d_7_67373*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_665252"
 conv2d_7/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_665352!
max_pooling2d_7/PartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0batch_normalization_7_67377batch_normalization_7_67379batch_normalization_7_67381batch_normalization_7_67383*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_665542/
-batch_normalization_7/StatefulPartitionedCall?
dropout_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_665692
dropout_7/PartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0conv2d_8_67387conv2d_8_67389*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_665822"
 conv2d_8/StatefulPartitionedCall?
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_665922!
max_pooling2d_8/PartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0batch_normalization_8_67393batch_normalization_8_67395batch_normalization_8_67397batch_normalization_8_67399*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_666112/
-batch_normalization_8/StatefulPartitionedCall?
dropout_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_666262
dropout_8/PartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0conv2d_9_67403conv2d_9_67405*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_666392"
 conv2d_9/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_9_67408batch_normalization_9_67410batch_normalization_9_67412batch_normalization_9_67414*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_666622/
-batch_normalization_9/StatefulPartitionedCall?
max_pooling2d_9/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_666762!
max_pooling2d_9/PartitionedCall?
dropout_9/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_666832
dropout_9/PartitionedCall?
*global_average_pooling2d_1/PartitionedCallPartitionedCall"dropout_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *^
fYRW
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_666902,
*global_average_pooling2d_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0dense_1_67420dense_1_67422*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_667022!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_6
?
b
)__inference_dropout_9_layer_call_fn_69888

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_667992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69618

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69564

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_66676

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_66881

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????  ?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????  ?*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????  ?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????  ?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????  ?2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
E
)__inference_dropout_9_layer_call_fn_69883

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_666832
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_66662

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
(__inference_conv2d_6_layer_call_fn_69144

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_664682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_model_2_layer_call_fn_67352
input_6!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:
??

unknown_24:	?
identity??StatefulPartitionedCall?
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
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_672402
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_6
?	
?
5__inference_batch_normalization_6_layer_call_fn_69275

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_664972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
??
?
__inference__traced_save_70160
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop5
1savev2_adam_conv2d_8_kernel_m_read_readvariableop3
/savev2_adam_conv2d_8_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_m_read_readvariableop5
1savev2_adam_conv2d_9_kernel_m_read_readvariableop3
/savev2_adam_conv2d_9_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop5
1savev2_adam_conv2d_8_kernel_v_read_readvariableop3
/savev2_adam_conv2d_8_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_8_beta_v_read_readvariableop5
1savev2_adam_conv2d_9_kernel_v_read_readvariableop3
/savev2_adam_conv2d_9_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_9_beta_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*?
value?B?FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop1savev2_adam_conv2d_8_kernel_m_read_readvariableop/savev2_adam_conv2d_8_bias_m_read_readvariableop=savev2_adam_batch_normalization_8_gamma_m_read_readvariableop<savev2_adam_batch_normalization_8_beta_m_read_readvariableop1savev2_adam_conv2d_9_kernel_m_read_readvariableop/savev2_adam_conv2d_9_bias_m_read_readvariableop=savev2_adam_batch_normalization_9_gamma_m_read_readvariableop<savev2_adam_batch_normalization_9_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop1savev2_adam_conv2d_8_kernel_v_read_readvariableop/savev2_adam_conv2d_8_bias_v_read_readvariableop=savev2_adam_batch_normalization_8_gamma_v_read_readvariableop<savev2_adam_batch_normalization_8_beta_v_read_readvariableop1savev2_adam_conv2d_9_kernel_v_read_readvariableop/savev2_adam_conv2d_9_bias_v_read_readvariableop=savev2_adam_batch_normalization_9_gamma_v_read_readvariableop<savev2_adam_batch_normalization_9_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *T
dtypesJ
H2F	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :@:@:@:@:@:@:@?:?:?:?:?:?:??:?:?:?:?:?:??:?:?:?:?:?:
??:?: : :@:@:@:@:@?:?:?:?:??:?:?:?:??:?:?:?:
??:?:@:@:@:@:@?:?:?:?:??:?:?:?:??:?:?:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@: 
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
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?: 

_output_shapes
: :!

_output_shapes
: :,"(
&
_output_shapes
:@: #

_output_shapes
:@: $

_output_shapes
:@: %

_output_shapes
:@:-&)
'
_output_shapes
:@?:!'

_output_shapes	
:?:!(

_output_shapes	
:?:!)

_output_shapes	
:?:.**
(
_output_shapes
:??:!+

_output_shapes	
:?:!,

_output_shapes	
:?:!-

_output_shapes	
:?:..*
(
_output_shapes
:??:!/

_output_shapes	
:?:!0

_output_shapes	
:?:!1

_output_shapes	
:?:&2"
 
_output_shapes
:
??:!3

_output_shapes	
:?:,4(
&
_output_shapes
:@: 5

_output_shapes
:@: 6

_output_shapes
:@: 7

_output_shapes
:@:-8)
'
_output_shapes
:@?:!9

_output_shapes	
:?:!:

_output_shapes	
:?:!;

_output_shapes	
:?:.<*
(
_output_shapes
:??:!=

_output_shapes	
:?:!>

_output_shapes	
:?:!?

_output_shapes	
:?:.@*
(
_output_shapes
:??:!A

_output_shapes	
:?:!B

_output_shapes	
:?:!C

_output_shapes	
:?:&D"
 
_output_shapes
:
??:!E

_output_shapes	
:?:F

_output_shapes
: 
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_66999

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_66525

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2	
BiasAddc
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
Relux
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
E
)__inference_dropout_6_layer_call_fn_69310

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_665122
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?^
?
B__inference_model_2_layer_call_and_return_conditional_losses_67500
input_6(
conv2d_6_67429:@
conv2d_6_67431:@)
batch_normalization_6_67435:@)
batch_normalization_6_67437:@)
batch_normalization_6_67439:@)
batch_normalization_6_67441:@)
conv2d_7_67445:@?
conv2d_7_67447:	?*
batch_normalization_7_67451:	?*
batch_normalization_7_67453:	?*
batch_normalization_7_67455:	?*
batch_normalization_7_67457:	?*
conv2d_8_67461:??
conv2d_8_67463:	?*
batch_normalization_8_67467:	?*
batch_normalization_8_67469:	?*
batch_normalization_8_67471:	?*
batch_normalization_8_67473:	?*
conv2d_9_67477:??
conv2d_9_67479:	?*
batch_normalization_9_67482:	?*
batch_normalization_9_67484:	?*
batch_normalization_9_67486:	?*
batch_normalization_9_67488:	?!
dense_1_67494:
??
dense_1_67496:	?
identity??-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_6conv2d_6_67429conv2d_6_67431*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_664682"
 conv2d_6/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_664782!
max_pooling2d_6/PartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0batch_normalization_6_67435batch_normalization_6_67437batch_normalization_6_67439batch_normalization_6_67441*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_670812/
-batch_normalization_6/StatefulPartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_670452#
!dropout_6/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0conv2d_7_67445conv2d_7_67447*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_665252"
 conv2d_7/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_665352!
max_pooling2d_7/PartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0batch_normalization_7_67451batch_normalization_7_67453batch_normalization_7_67455batch_normalization_7_67457*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_669992/
-batch_normalization_7/StatefulPartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_669632#
!dropout_7/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0conv2d_8_67461conv2d_8_67463*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_665822"
 conv2d_8/StatefulPartitionedCall?
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_665922!
max_pooling2d_8/PartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0batch_normalization_8_67467batch_normalization_8_67469batch_normalization_8_67471batch_normalization_8_67473*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_669172/
-batch_normalization_8/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_668812#
!dropout_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv2d_9_67477conv2d_9_67479*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_666392"
 conv2d_9/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_9_67482batch_normalization_9_67484batch_normalization_9_67486batch_normalization_9_67488*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_668402/
-batch_normalization_9/StatefulPartitionedCall?
max_pooling2d_9/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_666762!
max_pooling2d_9/PartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_667992#
!dropout_9/StatefulPartitionedCall?
*global_average_pooling2d_1/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *^
fYRW
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_666902,
*global_average_pooling2d_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0dense_1_67494dense_1_67496*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_667022!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_6
?
K
/__inference_max_pooling2d_8_layer_call_fn_69546

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_665922
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_8_layer_call_fn_69644

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_662182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_69135

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69200

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_6_layer_call_fn_69288

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_670812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69182

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_9_layer_call_fn_69815

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_663442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_66840

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_8_layer_call_fn_69541

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_661392
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
T
(__inference_lambda_1_layer_call_fn_69118
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_lambda_1_layer_call_and_return_conditional_losses_676032
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_66468

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_8_layer_call_fn_69670

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_669172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
b
)__inference_dropout_8_layer_call_fn_69697

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_668812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_69149

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
o
C__inference_lambda_1_layer_call_and_return_conditional_losses_69098
inputs_0
inputs_1
identityX
subSubinputs_0inputs_1*
T0*(
_output_shapes
:??????????2
subV
SquareSquaresub:z:0*
T0*(
_output_shapes
:??????????2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Sum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
	Maximum/yq
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constp
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
	Maximum_1U
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
(__inference_conv2d_8_layer_call_fn_69526

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_665822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_7_layer_call_fn_69350

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_659912
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_67081

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
(__inference_conv2d_9_layer_call_fn_69717

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_666392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
'__inference_model_2_layer_call_fn_69084

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:
??

unknown_24:	?
identity??StatefulPartitionedCall?
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
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_672402
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_7_layer_call_fn_69501

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_665692
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
B__inference_model_3_layer_call_and_return_conditional_losses_67835

inputs
inputs_1'
model_2_67753:@
model_2_67755:@
model_2_67757:@
model_2_67759:@
model_2_67761:@
model_2_67763:@(
model_2_67765:@?
model_2_67767:	?
model_2_67769:	?
model_2_67771:	?
model_2_67773:	?
model_2_67775:	?)
model_2_67777:??
model_2_67779:	?
model_2_67781:	?
model_2_67783:	?
model_2_67785:	?
model_2_67787:	?)
model_2_67789:??
model_2_67791:	?
model_2_67793:	?
model_2_67795:	?
model_2_67797:	?
model_2_67799:	?!
model_2_67801:
??
model_2_67803:	?
identity??model_2/StatefulPartitionedCall?!model_2/StatefulPartitionedCall_1?
model_2/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_2_67753model_2_67755model_2_67757model_2_67759model_2_67761model_2_67763model_2_67765model_2_67767model_2_67769model_2_67771model_2_67773model_2_67775model_2_67777model_2_67779model_2_67781model_2_67783model_2_67785model_2_67787model_2_67789model_2_67791model_2_67793model_2_67795model_2_67797model_2_67799model_2_67801model_2_67803*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_672402!
model_2/StatefulPartitionedCall?
!model_2/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_2_67753model_2_67755model_2_67757model_2_67759model_2_67761model_2_67763model_2_67765model_2_67767model_2_67769model_2_67771model_2_67773model_2_67775model_2_67777model_2_67779model_2_67781model_2_67783model_2_67785model_2_67787model_2_67789model_2_67791model_2_67793model_2_67795model_2_67797model_2_67799model_2_67801model_2_67803 ^model_2/StatefulPartitionedCall*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_672402#
!model_2/StatefulPartitionedCall_1?
lambda_1/PartitionedCallPartitionedCall(model_2/StatefulPartitionedCall:output:0*model_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_lambda_1_layer_call_and_return_conditional_losses_676852
lambda_1/PartitionedCall|
IdentityIdentity!lambda_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^model_2/StatefulPartitionedCall"^model_2/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall2F
!model_2/StatefulPartitionedCall_1!model_2/StatefulPartitionedCall_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_69154

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2	
MaxPooln
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_69846

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_68970

inputsA
'conv2d_6_conv2d_readvariableop_resource:@6
(conv2d_6_biasadd_readvariableop_resource:@;
-batch_normalization_6_readvariableop_resource:@=
/batch_normalization_6_readvariableop_1_resource:@L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_7_conv2d_readvariableop_resource:@?7
(conv2d_7_biasadd_readvariableop_resource:	?<
-batch_normalization_7_readvariableop_resource:	?>
/batch_normalization_7_readvariableop_1_resource:	?M
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	?C
'conv2d_8_conv2d_readvariableop_resource:??7
(conv2d_8_biasadd_readvariableop_resource:	?<
-batch_normalization_8_readvariableop_resource:	?>
/batch_normalization_8_readvariableop_1_resource:	?M
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	?C
'conv2d_9_conv2d_readvariableop_resource:??7
(conv2d_9_biasadd_readvariableop_resource:	?<
-batch_normalization_9_readvariableop_resource:	?>
/batch_normalization_9_readvariableop_1_resource:	?M
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	?O
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?
identity??$batch_normalization_6/AssignNewValue?&batch_normalization_6/AssignNewValue_1?5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_6/ReadVariableOp?&batch_normalization_6/ReadVariableOp_1?$batch_normalization_7/AssignNewValue?&batch_normalization_7/AssignNewValue_1?5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_7/ReadVariableOp?&batch_normalization_7/ReadVariableOp_1?$batch_normalization_8/AssignNewValue?&batch_normalization_8/AssignNewValue_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1?$batch_normalization_9/AssignNewValue?&batch_normalization_9/AssignNewValue_1?5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_9/ReadVariableOp?&batch_normalization_9/ReadVariableOp_1?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv2d_6/BiasAdd}
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
conv2d_6/Relu?
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool?
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp?
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1?
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_6/MaxPool:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_6/FusedBatchNormV3?
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue?
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1w
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_6/dropout/Const?
dropout_6/dropout/MulMul*batch_normalization_6/FusedBatchNormV3:y:0 dropout_6/dropout/Const:output:0*
T0*1
_output_shapes
:???????????@2
dropout_6/dropout/Mul?
dropout_6/dropout/ShapeShape*batch_normalization_6/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_6/dropout/Shape?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype0*

seedY20
.dropout_6/dropout/random_uniform/RandomUniform?
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_6/dropout/GreaterEqual/y?
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2 
dropout_6/dropout/GreaterEqual?
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2
dropout_6/dropout/Cast?
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@2
dropout_6/dropout/Mul_1?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Ddropout_6/dropout/Mul_1:z:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????2
conv2d_7/BiasAdd~
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*2
_output_shapes 
:????????????2
conv2d_7/Relu?
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu:activations:0*0
_output_shapes
:?????????@@?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPool?
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_7/ReadVariableOp?
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_7/ReadVariableOp_1?
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_7/MaxPool:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_7/FusedBatchNormV3?
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue?
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1w
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_7/dropout/Const?
dropout_7/dropout/MulMul*batch_normalization_7/FusedBatchNormV3:y:0 dropout_7/dropout/Const:output:0*
T0*0
_output_shapes
:?????????@@?2
dropout_7/dropout/Mul?
dropout_7/dropout/ShapeShape*batch_normalization_7/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_7/dropout/Shape?
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????@@?*
dtype0*

seedY*
seed220
.dropout_7/dropout/random_uniform/RandomUniform?
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_7/dropout/GreaterEqual/y?
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????@@?2 
dropout_7/dropout/GreaterEqual?
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????@@?2
dropout_7/dropout/Cast?
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????@@?2
dropout_7/dropout/Mul_1?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Ddropout_7/dropout/Mul_1:z:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2
conv2d_8/BiasAdd|
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
conv2d_8/Relu?
max_pooling2d_8/MaxPoolMaxPoolconv2d_8/Relu:activations:0*0
_output_shapes
:?????????  ?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_8/MaxPool?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_8/ReadVariableOp?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_8/ReadVariableOp_1?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_8/MaxPool:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_8/FusedBatchNormV3?
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue?
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1w
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_8/dropout/Const?
dropout_8/dropout/MulMul*batch_normalization_8/FusedBatchNormV3:y:0 dropout_8/dropout/Const:output:0*
T0*0
_output_shapes
:?????????  ?2
dropout_8/dropout/Mul?
dropout_8/dropout/ShapeShape*batch_normalization_8/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????  ?*
dtype0*

seedY*
seed220
.dropout_8/dropout/random_uniform/RandomUniform?
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_8/dropout/GreaterEqual/y?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????  ?2 
dropout_8/dropout/GreaterEqual?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????  ?2
dropout_8/dropout/Cast?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????  ?2
dropout_8/dropout/Mul_1?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2DConv2Ddropout_8/dropout/Mul_1:z:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv2d_9/Conv2D?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2
conv2d_9/BiasAdd|
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?2
conv2d_9/Relu?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$batch_normalization_9/ReadVariableOp?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&batch_normalization_9/ReadVariableOp_1?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_9/Relu:activations:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_9/FusedBatchNormV3?
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_9/AssignNewValue?
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_9/AssignNewValue_1?
max_pooling2d_9/MaxPoolMaxPool*batch_normalization_9/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_9/MaxPoolw
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_9/dropout/Const?
dropout_9/dropout/MulMul max_pooling2d_9/MaxPool:output:0 dropout_9/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout_9/dropout/Mul?
dropout_9/dropout/ShapeShape max_pooling2d_9/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0*

seedY*
seed220
.dropout_9/dropout/random_uniform/RandomUniform?
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_9/dropout/GreaterEqual/y?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2 
dropout_9/dropout/GreaterEqual?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout_9/dropout/Cast?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout_9/dropout/Mul_1?
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_1/Mean/reduction_indices?
global_average_pooling2d_1/MeanMeandropout_9/dropout/Mul_1:z:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2!
global_average_pooling2d_1/Mean?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMul(global_average_pooling2d_1/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddt
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_66582

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
B__inference_model_3_layer_call_and_return_conditional_losses_67606

inputs
inputs_1'
model_2_67509:@
model_2_67511:@
model_2_67513:@
model_2_67515:@
model_2_67517:@
model_2_67519:@(
model_2_67521:@?
model_2_67523:	?
model_2_67525:	?
model_2_67527:	?
model_2_67529:	?
model_2_67531:	?)
model_2_67533:??
model_2_67535:	?
model_2_67537:	?
model_2_67539:	?
model_2_67541:	?
model_2_67543:	?)
model_2_67545:??
model_2_67547:	?
model_2_67549:	?
model_2_67551:	?
model_2_67553:	?
model_2_67555:	?!
model_2_67557:
??
model_2_67559:	?
identity??model_2/StatefulPartitionedCall?!model_2/StatefulPartitionedCall_1?
model_2/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_2_67509model_2_67511model_2_67513model_2_67515model_2_67517model_2_67519model_2_67521model_2_67523model_2_67525model_2_67527model_2_67529model_2_67531model_2_67533model_2_67535model_2_67537model_2_67539model_2_67541model_2_67543model_2_67545model_2_67547model_2_67549model_2_67551model_2_67553model_2_67555model_2_67557model_2_67559*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_667092!
model_2/StatefulPartitionedCall?
!model_2/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_2_67509model_2_67511model_2_67513model_2_67515model_2_67517model_2_67519model_2_67521model_2_67523model_2_67525model_2_67527model_2_67529model_2_67531model_2_67533model_2_67535model_2_67537model_2_67539model_2_67541model_2_67543model_2_67545model_2_67547model_2_67549model_2_67551model_2_67553model_2_67555model_2_67557model_2_67559*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_667092#
!model_2/StatefulPartitionedCall_1?
lambda_1/PartitionedCallPartitionedCall(model_2/StatefulPartitionedCall:output:0*model_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_lambda_1_layer_call_and_return_conditional_losses_676032
lambda_1/PartitionedCall|
IdentityIdentity!lambda_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^model_2/StatefulPartitionedCall"^model_2/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall2F
!model_2/StatefulPartitionedCall_1!model_2/StatefulPartitionedCall_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_66070

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?^
?
B__inference_model_2_layer_call_and_return_conditional_losses_67240

inputs(
conv2d_6_67169:@
conv2d_6_67171:@)
batch_normalization_6_67175:@)
batch_normalization_6_67177:@)
batch_normalization_6_67179:@)
batch_normalization_6_67181:@)
conv2d_7_67185:@?
conv2d_7_67187:	?*
batch_normalization_7_67191:	?*
batch_normalization_7_67193:	?*
batch_normalization_7_67195:	?*
batch_normalization_7_67197:	?*
conv2d_8_67201:??
conv2d_8_67203:	?*
batch_normalization_8_67207:	?*
batch_normalization_8_67209:	?*
batch_normalization_8_67211:	?*
batch_normalization_8_67213:	?*
conv2d_9_67217:??
conv2d_9_67219:	?*
batch_normalization_9_67222:	?*
batch_normalization_9_67224:	?*
batch_normalization_9_67226:	?*
batch_normalization_9_67228:	?!
dense_1_67234:
??
dense_1_67236:	?
identity??-batch_normalization_6/StatefulPartitionedCall?-batch_normalization_7/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?!dropout_7/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_67169conv2d_6_67171*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_664682"
 conv2d_6/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_664782!
max_pooling2d_6/PartitionedCall?
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0batch_normalization_6_67175batch_normalization_6_67177batch_normalization_6_67179batch_normalization_6_67181*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_670812/
-batch_normalization_6/StatefulPartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_670452#
!dropout_6/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0conv2d_7_67185conv2d_7_67187*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_665252"
 conv2d_7/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_665352!
max_pooling2d_7/PartitionedCall?
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0batch_normalization_7_67191batch_normalization_7_67193batch_normalization_7_67195batch_normalization_7_67197*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_669992/
-batch_normalization_7/StatefulPartitionedCall?
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_669632#
!dropout_7/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0conv2d_8_67201conv2d_8_67203*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_665822"
 conv2d_8/StatefulPartitionedCall?
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_665922!
max_pooling2d_8/PartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0batch_normalization_8_67207batch_normalization_8_67209batch_normalization_8_67211batch_normalization_8_67213*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_669172/
-batch_normalization_8/StatefulPartitionedCall?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_668812#
!dropout_8/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv2d_9_67217conv2d_9_67219*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_666392"
 conv2d_9/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_9_67222batch_normalization_9_67224batch_normalization_9_67226batch_normalization_9_67228*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_668402/
-batch_normalization_9/StatefulPartitionedCall?
max_pooling2d_9/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_666762!
max_pooling2d_9/PartitionedCall?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_9_layer_call_and_return_conditional_losses_667992#
!dropout_9/StatefulPartitionedCall?
*global_average_pooling2d_1/PartitionedCallPartitionedCall*dropout_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *^
fYRW
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_666902,
*global_average_pooling2d_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0dense_1_67234dense_1_67236*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_667022!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_66702

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_67661
input_4
input_5!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:
??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4input_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_676062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_5
?
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_66413

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_9_layer_call_and_return_conditional_losses_69866

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_69929

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_667022
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_68676
inputs_0
inputs_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:
??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_676062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapesp
n:???????????:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69753

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_69675

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????  ?2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????  ?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_65991

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_69851

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_66569

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????@@?2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????@@?2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_6_layer_call_fn_69249

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_658782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
'__inference_model_2_layer_call_fn_66764
input_6!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?&

unknown_17:??

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:
??

unknown_24:	?
identity??StatefulPartitionedCall?
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
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_667092
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_6
?
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_69293

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????@2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
c
D__inference_dropout_6_layer_call_and_return_conditional_losses_69305

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
m
C__inference_lambda_1_layer_call_and_return_conditional_losses_67685

inputs
inputs_1
identityV
subSubinputsinputs_1*
T0*(
_output_shapes
:??????????2
subV
SquareSquaresub:z:0*
T0*(
_output_shapes
:??????????2
Squarep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
Sum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
	Maximum/yq
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:?????????2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Constp
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
	Maximum_1U
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69236

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_8_layer_call_fn_69631

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_661742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_7_layer_call_fn_69506

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_669632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_9_layer_call_fn_69802

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_663002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_7_layer_call_fn_69335

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_665252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_66478

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
2	
MaxPooln
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_6_layer_call_fn_69159

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_658432
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_69708

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????  ?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_66611

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_6_layer_call_fn_69164

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_664782
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69582

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_9_layer_call_and_return_conditional_losses_69878

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_6_layer_call_fn_69315

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_670452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
c
D__inference_dropout_6_layer_call_and_return_conditional_losses_67045

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_69517

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@@?2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_66300

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_4:
serving_default_input_4:0???????????
E
input_5:
serving_default_input_5:0???????????<
lambda_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
layer-8
layer_with_weights-4
layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
	variables
regularization_losses
 trainable_variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_network
?
"	variables
#regularization_losses
$trainable_variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
&iter

'beta_1

(beta_2
	)decay
*learning_rate+m?,m?-m?.m?1m?2m?3m?4m?7m?8m?9m?:m?=m?>m??m?@m?Cm?Dm?+v?,v?-v?.v?1v?2v?3v?4v?7v?8v?9v?:v?=v?>v??v?@v?Cv?Dv?"
	optimizer
?
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25"
trackable_list_wrapper
 "
trackable_list_wrapper
?
+0
,1
-2
.3
14
25
36
47
78
89
910
:11
=12
>13
?14
@15
C16
D17"
trackable_list_wrapper
?
Elayer_metrics

Flayers
Glayer_regularization_losses
	variables
Hnon_trainable_variables
Imetrics
regularization_losses
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
"
_tf_keras_input_layer
?

+kernel
,bias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Raxis
	-gamma
.beta
/moving_mean
0moving_variance
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

1kernel
2bias
[	variables
\regularization_losses
]trainable_variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
_	variables
`regularization_losses
atrainable_variables
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
caxis
	3gamma
4beta
5moving_mean
6moving_variance
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

7kernel
8bias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
taxis
	9gamma
:beta
;moving_mean
<moving_variance
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
y	variables
zregularization_losses
{trainable_variables
|	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

=kernel
>bias
}	variables
~regularization_losses
trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Ckernel
Dbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25"
trackable_list_wrapper
 "
trackable_list_wrapper
?
+0
,1
-2
.3
14
25
36
47
78
89
910
:11
=12
>13
?14
@15
C16
D17"
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
	variables
?non_trainable_variables
?metrics
regularization_losses
 trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
"	variables
?non_trainable_variables
?metrics
#regularization_losses
$trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'@2conv2d_6/kernel
:@2conv2d_6/bias
):'@2batch_normalization_6/gamma
(:&@2batch_normalization_6/beta
1:/@ (2!batch_normalization_6/moving_mean
5:3@ (2%batch_normalization_6/moving_variance
*:(@?2conv2d_7/kernel
:?2conv2d_7/bias
*:(?2batch_normalization_7/gamma
):'?2batch_normalization_7/beta
2:0? (2!batch_normalization_7/moving_mean
6:4? (2%batch_normalization_7/moving_variance
+:)??2conv2d_8/kernel
:?2conv2d_8/bias
*:(?2batch_normalization_8/gamma
):'?2batch_normalization_8/beta
2:0? (2!batch_normalization_8/moving_mean
6:4? (2%batch_normalization_8/moving_variance
+:)??2conv2d_9/kernel
:?2conv2d_9/bias
*:(?2batch_normalization_9/gamma
):'?2batch_normalization_9/beta
2:0? (2!batch_normalization_9/moving_mean
6:4? (2%batch_normalization_9/moving_variance
": 
??2dense_1/kernel
:?2dense_1/bias
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
X
/0
01
52
63
;4
<5
A6
B7"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
J	variables
?non_trainable_variables
?metrics
Kregularization_losses
Ltrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
N	variables
?non_trainable_variables
?metrics
Oregularization_losses
Ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
-0
.1
/2
03"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
S	variables
?non_trainable_variables
?metrics
Tregularization_losses
Utrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
W	variables
?non_trainable_variables
?metrics
Xregularization_losses
Ytrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
[	variables
?non_trainable_variables
?metrics
\regularization_losses
]trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
_	variables
?non_trainable_variables
?metrics
`regularization_losses
atrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
d	variables
?non_trainable_variables
?metrics
eregularization_losses
ftrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
h	variables
?non_trainable_variables
?metrics
iregularization_losses
jtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
l	variables
?non_trainable_variables
?metrics
mregularization_losses
ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
p	variables
?non_trainable_variables
?metrics
qregularization_losses
rtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
90
:1
;2
<3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
u	variables
?non_trainable_variables
?metrics
vregularization_losses
wtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
y	variables
?non_trainable_variables
?metrics
zregularization_losses
{trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
}	variables
?non_trainable_variables
?metrics
~regularization_losses
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
?regularization_losses
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
?regularization_losses
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
?regularization_losses
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
?regularization_losses
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
?layer_metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
?regularization_losses
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
 "
trackable_list_wrapper
X
/0
01
52
63
;4
<5
A6
B7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
.:,@2Adam/conv2d_6/kernel/m
 :@2Adam/conv2d_6/bias/m
.:,@2"Adam/batch_normalization_6/gamma/m
-:+@2!Adam/batch_normalization_6/beta/m
/:-@?2Adam/conv2d_7/kernel/m
!:?2Adam/conv2d_7/bias/m
/:-?2"Adam/batch_normalization_7/gamma/m
.:,?2!Adam/batch_normalization_7/beta/m
0:.??2Adam/conv2d_8/kernel/m
!:?2Adam/conv2d_8/bias/m
/:-?2"Adam/batch_normalization_8/gamma/m
.:,?2!Adam/batch_normalization_8/beta/m
0:.??2Adam/conv2d_9/kernel/m
!:?2Adam/conv2d_9/bias/m
/:-?2"Adam/batch_normalization_9/gamma/m
.:,?2!Adam/batch_normalization_9/beta/m
':%
??2Adam/dense_1/kernel/m
 :?2Adam/dense_1/bias/m
.:,@2Adam/conv2d_6/kernel/v
 :@2Adam/conv2d_6/bias/v
.:,@2"Adam/batch_normalization_6/gamma/v
-:+@2!Adam/batch_normalization_6/beta/v
/:-@?2Adam/conv2d_7/kernel/v
!:?2Adam/conv2d_7/bias/v
/:-?2"Adam/batch_normalization_7/gamma/v
.:,?2!Adam/batch_normalization_7/beta/v
0:.??2Adam/conv2d_8/kernel/v
!:?2Adam/conv2d_8/bias/v
/:-?2"Adam/batch_normalization_8/gamma/v
.:,?2!Adam/batch_normalization_8/beta/v
0:.??2Adam/conv2d_9/kernel/v
!:?2Adam/conv2d_9/bias/v
/:-?2"Adam/batch_normalization_9/gamma/v
.:,?2!Adam/batch_normalization_9/beta/v
':%
??2Adam/dense_1/kernel/v
 :?2Adam/dense_1/bias/v
?B?
 __inference__wrapped_model_65834input_4input_5"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_model_3_layer_call_and_return_conditional_losses_68374
B__inference_model_3_layer_call_and_return_conditional_losses_68618
B__inference_model_3_layer_call_and_return_conditional_losses_68034
B__inference_model_3_layer_call_and_return_conditional_losses_68120?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_model_3_layer_call_fn_67661
'__inference_model_3_layer_call_fn_68676
'__inference_model_3_layer_call_fn_68734
'__inference_model_3_layer_call_fn_67948?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_2_layer_call_and_return_conditional_losses_68838
B__inference_model_2_layer_call_and_return_conditional_losses_68970
B__inference_model_2_layer_call_and_return_conditional_losses_67426
B__inference_model_2_layer_call_and_return_conditional_losses_67500?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_model_2_layer_call_fn_66764
'__inference_model_2_layer_call_fn_69027
'__inference_model_2_layer_call_fn_69084
'__inference_model_2_layer_call_fn_67352?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_lambda_1_layer_call_and_return_conditional_losses_69098
C__inference_lambda_1_layer_call_and_return_conditional_losses_69112?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_lambda_1_layer_call_fn_69118
(__inference_lambda_1_layer_call_fn_69124?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_68186input_4input_5"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_6_layer_call_and_return_conditional_losses_69135?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_6_layer_call_fn_69144?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_69149
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_69154?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling2d_6_layer_call_fn_69159
/__inference_max_pooling2d_6_layer_call_fn_69164?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69182
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69200
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69218
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69236?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_6_layer_call_fn_69249
5__inference_batch_normalization_6_layer_call_fn_69262
5__inference_batch_normalization_6_layer_call_fn_69275
5__inference_batch_normalization_6_layer_call_fn_69288?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_6_layer_call_and_return_conditional_losses_69293
D__inference_dropout_6_layer_call_and_return_conditional_losses_69305?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_6_layer_call_fn_69310
)__inference_dropout_6_layer_call_fn_69315?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_69326?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_7_layer_call_fn_69335?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_69340
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_69345?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling2d_7_layer_call_fn_69350
/__inference_max_pooling2d_7_layer_call_fn_69355?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69373
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69391
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69409
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69427?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_7_layer_call_fn_69440
5__inference_batch_normalization_7_layer_call_fn_69453
5__inference_batch_normalization_7_layer_call_fn_69466
5__inference_batch_normalization_7_layer_call_fn_69479?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_7_layer_call_and_return_conditional_losses_69484
D__inference_dropout_7_layer_call_and_return_conditional_losses_69496?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_7_layer_call_fn_69501
)__inference_dropout_7_layer_call_fn_69506?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_69517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_8_layer_call_fn_69526?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_69531
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_69536?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling2d_8_layer_call_fn_69541
/__inference_max_pooling2d_8_layer_call_fn_69546?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69564
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69582
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69600
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69618?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_8_layer_call_fn_69631
5__inference_batch_normalization_8_layer_call_fn_69644
5__inference_batch_normalization_8_layer_call_fn_69657
5__inference_batch_normalization_8_layer_call_fn_69670?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_8_layer_call_and_return_conditional_losses_69675
D__inference_dropout_8_layer_call_and_return_conditional_losses_69687?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_8_layer_call_fn_69692
)__inference_dropout_8_layer_call_fn_69697?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_69708?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_9_layer_call_fn_69717?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69735
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69753
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69771
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69789?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_9_layer_call_fn_69802
5__inference_batch_normalization_9_layer_call_fn_69815
5__inference_batch_normalization_9_layer_call_fn_69828
5__inference_batch_normalization_9_layer_call_fn_69841?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_69846
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_69851?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling2d_9_layer_call_fn_69856
/__inference_max_pooling2d_9_layer_call_fn_69861?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dropout_9_layer_call_and_return_conditional_losses_69866
D__inference_dropout_9_layer_call_and_return_conditional_losses_69878?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_9_layer_call_fn_69883
)__inference_dropout_9_layer_call_fn_69888?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_69894
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_69900?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
:__inference_global_average_pooling2d_1_layer_call_fn_69905
:__inference_global_average_pooling2d_1_layer_call_fn_69910?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_69920?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_69929?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_65834?+,-./0123456789:;<=>?@ABCDl?i
b?_
]?Z
+?(
input_4???????????
+?(
input_5???????????
? "3?0
.
lambda_1"?
lambda_1??????????
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69182?-./0M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69200?-./0M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69218v-./0=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_69236v-./0=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
5__inference_batch_normalization_6_layer_call_fn_69249?-./0M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
5__inference_batch_normalization_6_layer_call_fn_69262?-./0M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
5__inference_batch_normalization_6_layer_call_fn_69275i-./0=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
5__inference_batch_normalization_6_layer_call_fn_69288i-./0=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69373?3456N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69391?3456N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69409t3456<?9
2?/
)?&
inputs?????????@@?
p 
? ".?+
$?!
0?????????@@?
? ?
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_69427t3456<?9
2?/
)?&
inputs?????????@@?
p
? ".?+
$?!
0?????????@@?
? ?
5__inference_batch_normalization_7_layer_call_fn_69440?3456N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
5__inference_batch_normalization_7_layer_call_fn_69453?3456N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
5__inference_batch_normalization_7_layer_call_fn_69466g3456<?9
2?/
)?&
inputs?????????@@?
p 
? "!??????????@@??
5__inference_batch_normalization_7_layer_call_fn_69479g3456<?9
2?/
)?&
inputs?????????@@?
p
? "!??????????@@??
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69564?9:;<N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69582?9:;<N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69600t9:;<<?9
2?/
)?&
inputs?????????  ?
p 
? ".?+
$?!
0?????????  ?
? ?
P__inference_batch_normalization_8_layer_call_and_return_conditional_losses_69618t9:;<<?9
2?/
)?&
inputs?????????  ?
p
? ".?+
$?!
0?????????  ?
? ?
5__inference_batch_normalization_8_layer_call_fn_69631?9:;<N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
5__inference_batch_normalization_8_layer_call_fn_69644?9:;<N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
5__inference_batch_normalization_8_layer_call_fn_69657g9:;<<?9
2?/
)?&
inputs?????????  ?
p 
? "!??????????  ??
5__inference_batch_normalization_8_layer_call_fn_69670g9:;<<?9
2?/
)?&
inputs?????????  ?
p
? "!??????????  ??
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69735??@ABN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69753??@ABN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69771t?@AB<?9
2?/
)?&
inputs?????????  ?
p 
? ".?+
$?!
0?????????  ?
? ?
P__inference_batch_normalization_9_layer_call_and_return_conditional_losses_69789t?@AB<?9
2?/
)?&
inputs?????????  ?
p
? ".?+
$?!
0?????????  ?
? ?
5__inference_batch_normalization_9_layer_call_fn_69802??@ABN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
5__inference_batch_normalization_9_layer_call_fn_69815??@ABN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
5__inference_batch_normalization_9_layer_call_fn_69828g?@AB<?9
2?/
)?&
inputs?????????  ?
p 
? "!??????????  ??
5__inference_batch_normalization_9_layer_call_fn_69841g?@AB<?9
2?/
)?&
inputs?????????  ?
p
? "!??????????  ??
C__inference_conv2d_6_layer_call_and_return_conditional_losses_69135p+,9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????@
? ?
(__inference_conv2d_6_layer_call_fn_69144c+,9?6
/?,
*?'
inputs???????????
? ""????????????@?
C__inference_conv2d_7_layer_call_and_return_conditional_losses_69326q129?6
/?,
*?'
inputs???????????@
? "0?-
&?#
0????????????
? ?
(__inference_conv2d_7_layer_call_fn_69335d129?6
/?,
*?'
inputs???????????@
? "#? ?????????????
C__inference_conv2d_8_layer_call_and_return_conditional_losses_69517n788?5
.?+
)?&
inputs?????????@@?
? ".?+
$?!
0?????????@@?
? ?
(__inference_conv2d_8_layer_call_fn_69526a788?5
.?+
)?&
inputs?????????@@?
? "!??????????@@??
C__inference_conv2d_9_layer_call_and_return_conditional_losses_69708n=>8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
(__inference_conv2d_9_layer_call_fn_69717a=>8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
B__inference_dense_1_layer_call_and_return_conditional_losses_69920^CD0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_dense_1_layer_call_fn_69929QCD0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dropout_6_layer_call_and_return_conditional_losses_69293p=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
D__inference_dropout_6_layer_call_and_return_conditional_losses_69305p=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
)__inference_dropout_6_layer_call_fn_69310c=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
)__inference_dropout_6_layer_call_fn_69315c=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
D__inference_dropout_7_layer_call_and_return_conditional_losses_69484n<?9
2?/
)?&
inputs?????????@@?
p 
? ".?+
$?!
0?????????@@?
? ?
D__inference_dropout_7_layer_call_and_return_conditional_losses_69496n<?9
2?/
)?&
inputs?????????@@?
p
? ".?+
$?!
0?????????@@?
? ?
)__inference_dropout_7_layer_call_fn_69501a<?9
2?/
)?&
inputs?????????@@?
p 
? "!??????????@@??
)__inference_dropout_7_layer_call_fn_69506a<?9
2?/
)?&
inputs?????????@@?
p
? "!??????????@@??
D__inference_dropout_8_layer_call_and_return_conditional_losses_69675n<?9
2?/
)?&
inputs?????????  ?
p 
? ".?+
$?!
0?????????  ?
? ?
D__inference_dropout_8_layer_call_and_return_conditional_losses_69687n<?9
2?/
)?&
inputs?????????  ?
p
? ".?+
$?!
0?????????  ?
? ?
)__inference_dropout_8_layer_call_fn_69692a<?9
2?/
)?&
inputs?????????  ?
p 
? "!??????????  ??
)__inference_dropout_8_layer_call_fn_69697a<?9
2?/
)?&
inputs?????????  ?
p
? "!??????????  ??
D__inference_dropout_9_layer_call_and_return_conditional_losses_69866n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
D__inference_dropout_9_layer_call_and_return_conditional_losses_69878n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
)__inference_dropout_9_layer_call_fn_69883a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
)__inference_dropout_9_layer_call_fn_69888a<?9
2?/
)?&
inputs??????????
p
? "!????????????
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_69894?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_69900b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????
? ?
:__inference_global_average_pooling2d_1_layer_call_fn_69905wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
:__inference_global_average_pooling2d_1_layer_call_fn_69910U8?5
.?+
)?&
inputs??????????
? "????????????
C__inference_lambda_1_layer_call_and_return_conditional_losses_69098?d?a
Z?W
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????

 
p 
? "%?"
?
0?????????
? ?
C__inference_lambda_1_layer_call_and_return_conditional_losses_69112?d?a
Z?W
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????

 
p
? "%?"
?
0?????????
? ?
(__inference_lambda_1_layer_call_fn_69118?d?a
Z?W
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????

 
p 
? "???????????
(__inference_lambda_1_layer_call_fn_69124?d?a
Z?W
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????

 
p
? "???????????
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_69149?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_69154l9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
/__inference_max_pooling2d_6_layer_call_fn_69159?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
/__inference_max_pooling2d_6_layer_call_fn_69164_9?6
/?,
*?'
inputs???????????@
? ""????????????@?
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_69340?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_69345l:?7
0?-
+?(
inputs????????????
? ".?+
$?!
0?????????@@?
? ?
/__inference_max_pooling2d_7_layer_call_fn_69350?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
/__inference_max_pooling2d_7_layer_call_fn_69355_:?7
0?-
+?(
inputs????????????
? "!??????????@@??
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_69531?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_69536j8?5
.?+
)?&
inputs?????????@@?
? ".?+
$?!
0?????????  ?
? ?
/__inference_max_pooling2d_8_layer_call_fn_69541?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
/__inference_max_pooling2d_8_layer_call_fn_69546]8?5
.?+
)?&
inputs?????????@@?
? "!??????????  ??
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_69846?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_69851j8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0??????????
? ?
/__inference_max_pooling2d_9_layer_call_fn_69856?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
/__inference_max_pooling2d_9_layer_call_fn_69861]8?5
.?+
)?&
inputs?????????  ?
? "!????????????
B__inference_model_2_layer_call_and_return_conditional_losses_67426?+,-./0123456789:;<=>?@ABCDB??
8?5
+?(
input_6???????????
p 

 
? "&?#
?
0??????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_67500?+,-./0123456789:;<=>?@ABCDB??
8?5
+?(
input_6???????????
p

 
? "&?#
?
0??????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_68838?+,-./0123456789:;<=>?@ABCDA?>
7?4
*?'
inputs???????????
p 

 
? "&?#
?
0??????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_68970?+,-./0123456789:;<=>?@ABCDA?>
7?4
*?'
inputs???????????
p

 
? "&?#
?
0??????????
? ?
'__inference_model_2_layer_call_fn_66764{+,-./0123456789:;<=>?@ABCDB??
8?5
+?(
input_6???????????
p 

 
? "????????????
'__inference_model_2_layer_call_fn_67352{+,-./0123456789:;<=>?@ABCDB??
8?5
+?(
input_6???????????
p

 
? "????????????
'__inference_model_2_layer_call_fn_69027z+,-./0123456789:;<=>?@ABCDA?>
7?4
*?'
inputs???????????
p 

 
? "????????????
'__inference_model_2_layer_call_fn_69084z+,-./0123456789:;<=>?@ABCDA?>
7?4
*?'
inputs???????????
p

 
? "????????????
B__inference_model_3_layer_call_and_return_conditional_losses_68034?+,-./0123456789:;<=>?@ABCDt?q
j?g
]?Z
+?(
input_4???????????
+?(
input_5???????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_68120?+,-./0123456789:;<=>?@ABCDt?q
j?g
]?Z
+?(
input_4???????????
+?(
input_5???????????
p

 
? "%?"
?
0?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_68374?+,-./0123456789:;<=>?@ABCDv?s
l?i
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_68618?+,-./0123456789:;<=>?@ABCDv?s
l?i
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
p

 
? "%?"
?
0?????????
? ?
'__inference_model_3_layer_call_fn_67661?+,-./0123456789:;<=>?@ABCDt?q
j?g
]?Z
+?(
input_4???????????
+?(
input_5???????????
p 

 
? "???????????
'__inference_model_3_layer_call_fn_67948?+,-./0123456789:;<=>?@ABCDt?q
j?g
]?Z
+?(
input_4???????????
+?(
input_5???????????
p

 
? "???????????
'__inference_model_3_layer_call_fn_68676?+,-./0123456789:;<=>?@ABCDv?s
l?i
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
p 

 
? "???????????
'__inference_model_3_layer_call_fn_68734?+,-./0123456789:;<=>?@ABCDv?s
l?i
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
p

 
? "???????????
#__inference_signature_wrapper_68186?+,-./0123456789:;<=>?@ABCD}?z
? 
s?p
6
input_4+?(
input_4???????????
6
input_5+?(
input_5???????????"3?0
.
lambda_1"?
lambda_1?????????