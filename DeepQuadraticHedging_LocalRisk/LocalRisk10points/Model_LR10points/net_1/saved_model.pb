Ü
ø
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
.
Rsqrt
x"T
y"T"
Ttype:

2
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
H
ShardedFilename
basename	
shard

num_shards
filename
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02unknown8¾
¼
2feed_forward_sub_net_1/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_1/batch_normalization_6/gamma
µ
Ffeed_forward_sub_net_1/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_1/batch_normalization_6/gamma*
_output_shapes
:*
dtype0
º
1feed_forward_sub_net_1/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31feed_forward_sub_net_1/batch_normalization_6/beta
³
Efeed_forward_sub_net_1/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp1feed_forward_sub_net_1/batch_normalization_6/beta*
_output_shapes
:*
dtype0
¼
2feed_forward_sub_net_1/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_1/batch_normalization_7/gamma
µ
Ffeed_forward_sub_net_1/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_1/batch_normalization_7/gamma*
_output_shapes
:*
dtype0
º
1feed_forward_sub_net_1/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31feed_forward_sub_net_1/batch_normalization_7/beta
³
Efeed_forward_sub_net_1/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp1feed_forward_sub_net_1/batch_normalization_7/beta*
_output_shapes
:*
dtype0
¼
2feed_forward_sub_net_1/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_1/batch_normalization_8/gamma
µ
Ffeed_forward_sub_net_1/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_1/batch_normalization_8/gamma*
_output_shapes
:*
dtype0
º
1feed_forward_sub_net_1/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31feed_forward_sub_net_1/batch_normalization_8/beta
³
Efeed_forward_sub_net_1/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp1feed_forward_sub_net_1/batch_normalization_8/beta*
_output_shapes
:*
dtype0
¼
2feed_forward_sub_net_1/batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_1/batch_normalization_9/gamma
µ
Ffeed_forward_sub_net_1/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_1/batch_normalization_9/gamma*
_output_shapes
:*
dtype0
º
1feed_forward_sub_net_1/batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31feed_forward_sub_net_1/batch_normalization_9/beta
³
Efeed_forward_sub_net_1/batch_normalization_9/beta/Read/ReadVariableOpReadVariableOp1feed_forward_sub_net_1/batch_normalization_9/beta*
_output_shapes
:*
dtype0
¾
3feed_forward_sub_net_1/batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_1/batch_normalization_10/gamma
·
Gfeed_forward_sub_net_1/batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_1/batch_normalization_10/gamma*
_output_shapes
:*
dtype0
¼
2feed_forward_sub_net_1/batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_1/batch_normalization_10/beta
µ
Ffeed_forward_sub_net_1/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_1/batch_normalization_10/beta*
_output_shapes
:*
dtype0
¦
%feed_forward_sub_net_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%feed_forward_sub_net_1/dense_5/kernel

9feed_forward_sub_net_1/dense_5/kernel/Read/ReadVariableOpReadVariableOp%feed_forward_sub_net_1/dense_5/kernel*
_output_shapes

:*
dtype0
¦
%feed_forward_sub_net_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%feed_forward_sub_net_1/dense_6/kernel

9feed_forward_sub_net_1/dense_6/kernel/Read/ReadVariableOpReadVariableOp%feed_forward_sub_net_1/dense_6/kernel*
_output_shapes

:*
dtype0
¦
%feed_forward_sub_net_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%feed_forward_sub_net_1/dense_7/kernel

9feed_forward_sub_net_1/dense_7/kernel/Read/ReadVariableOpReadVariableOp%feed_forward_sub_net_1/dense_7/kernel*
_output_shapes

:*
dtype0
¦
%feed_forward_sub_net_1/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%feed_forward_sub_net_1/dense_8/kernel

9feed_forward_sub_net_1/dense_8/kernel/Read/ReadVariableOpReadVariableOp%feed_forward_sub_net_1/dense_8/kernel*
_output_shapes

:*
dtype0
¦
%feed_forward_sub_net_1/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%feed_forward_sub_net_1/dense_9/kernel

9feed_forward_sub_net_1/dense_9/kernel/Read/ReadVariableOpReadVariableOp%feed_forward_sub_net_1/dense_9/kernel*
_output_shapes

:*
dtype0

#feed_forward_sub_net_1/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#feed_forward_sub_net_1/dense_9/bias

7feed_forward_sub_net_1/dense_9/bias/Read/ReadVariableOpReadVariableOp#feed_forward_sub_net_1/dense_9/bias*
_output_shapes
:*
dtype0
È
8feed_forward_sub_net_1/batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8feed_forward_sub_net_1/batch_normalization_6/moving_mean
Á
Lfeed_forward_sub_net_1/batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp8feed_forward_sub_net_1/batch_normalization_6/moving_mean*
_output_shapes
:*
dtype0
Ð
<feed_forward_sub_net_1/batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><feed_forward_sub_net_1/batch_normalization_6/moving_variance
É
Pfeed_forward_sub_net_1/batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp<feed_forward_sub_net_1/batch_normalization_6/moving_variance*
_output_shapes
:*
dtype0
È
8feed_forward_sub_net_1/batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8feed_forward_sub_net_1/batch_normalization_7/moving_mean
Á
Lfeed_forward_sub_net_1/batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp8feed_forward_sub_net_1/batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
Ð
<feed_forward_sub_net_1/batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><feed_forward_sub_net_1/batch_normalization_7/moving_variance
É
Pfeed_forward_sub_net_1/batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp<feed_forward_sub_net_1/batch_normalization_7/moving_variance*
_output_shapes
:*
dtype0
È
8feed_forward_sub_net_1/batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8feed_forward_sub_net_1/batch_normalization_8/moving_mean
Á
Lfeed_forward_sub_net_1/batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp8feed_forward_sub_net_1/batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
Ð
<feed_forward_sub_net_1/batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><feed_forward_sub_net_1/batch_normalization_8/moving_variance
É
Pfeed_forward_sub_net_1/batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp<feed_forward_sub_net_1/batch_normalization_8/moving_variance*
_output_shapes
:*
dtype0
È
8feed_forward_sub_net_1/batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8feed_forward_sub_net_1/batch_normalization_9/moving_mean
Á
Lfeed_forward_sub_net_1/batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp8feed_forward_sub_net_1/batch_normalization_9/moving_mean*
_output_shapes
:*
dtype0
Ð
<feed_forward_sub_net_1/batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><feed_forward_sub_net_1/batch_normalization_9/moving_variance
É
Pfeed_forward_sub_net_1/batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp<feed_forward_sub_net_1/batch_normalization_9/moving_variance*
_output_shapes
:*
dtype0
Ê
9feed_forward_sub_net_1/batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_1/batch_normalization_10/moving_mean
Ã
Mfeed_forward_sub_net_1/batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_1/batch_normalization_10/moving_mean*
_output_shapes
:*
dtype0
Ò
=feed_forward_sub_net_1/batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_1/batch_normalization_10/moving_variance
Ë
Qfeed_forward_sub_net_1/batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_1/batch_normalization_10/moving_variance*
_output_shapes
:*
dtype0

NoOpNoOp
Ù;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*;
value;B; B;

	bn_layers
dense_layers
trainable_variables
	variables
regularization_losses
	keras_api

signatures
*
0
	1

2
3
4
5
#
0
1
2
3
4
v
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
Æ
0
1
2
3
4
5
6
7
8
9
#10
$11
%12
&13
'14
(15
)16
*17
+18
,19
20
21
22
 23
!24
"25
 
­
-layer_regularization_losses
.layer_metrics
/non_trainable_variables
trainable_variables
	variables

0layers
1metrics
regularization_losses
 

2axis
	gamma
beta
#moving_mean
$moving_variance
3trainable_variables
4	variables
5regularization_losses
6	keras_api

7axis
	gamma
beta
%moving_mean
&moving_variance
8trainable_variables
9	variables
:regularization_losses
;	keras_api

<axis
	gamma
beta
'moving_mean
(moving_variance
=trainable_variables
>	variables
?regularization_losses
@	keras_api

Aaxis
	gamma
beta
)moving_mean
*moving_variance
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api

Faxis
	gamma
beta
+moving_mean
,moving_variance
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api

K	keras_api
^

kernel
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
^

kernel
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
^

kernel
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
^

 kernel
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
h

!kernel
"bias
\trainable_variables
]	variables
^regularization_losses
_	keras_api
xv
VARIABLE_VALUE2feed_forward_sub_net_1/batch_normalization_6/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1feed_forward_sub_net_1/batch_normalization_6/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2feed_forward_sub_net_1/batch_normalization_7/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1feed_forward_sub_net_1/batch_normalization_7/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2feed_forward_sub_net_1/batch_normalization_8/gamma0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1feed_forward_sub_net_1/batch_normalization_8/beta0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2feed_forward_sub_net_1/batch_normalization_9/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1feed_forward_sub_net_1/batch_normalization_9/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE3feed_forward_sub_net_1/batch_normalization_10/gamma0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2feed_forward_sub_net_1/batch_normalization_10/beta0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%feed_forward_sub_net_1/dense_5/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%feed_forward_sub_net_1/dense_6/kernel1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%feed_forward_sub_net_1/dense_7/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%feed_forward_sub_net_1/dense_8/kernel1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%feed_forward_sub_net_1/dense_9/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#feed_forward_sub_net_1/dense_9/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8feed_forward_sub_net_1/batch_normalization_6/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE<feed_forward_sub_net_1/batch_normalization_6/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8feed_forward_sub_net_1/batch_normalization_7/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE<feed_forward_sub_net_1/batch_normalization_7/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8feed_forward_sub_net_1/batch_normalization_8/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE<feed_forward_sub_net_1/batch_normalization_8/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE8feed_forward_sub_net_1/batch_normalization_9/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE<feed_forward_sub_net_1/batch_normalization_9/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_1/batch_normalization_10/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_1/batch_normalization_10/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
 
 
F
#0
$1
%2
&3
'4
(5
)6
*7
+8
,9
N
0
	1

2
3
4
5
6
7
8
9
10
 
 

0
1

0
1
#2
$3
 
­
`layer_regularization_losses
alayer_metrics
bnon_trainable_variables
3trainable_variables
4	variables

clayers
dmetrics
5regularization_losses
 

0
1

0
1
%2
&3
 
­
elayer_regularization_losses
flayer_metrics
gnon_trainable_variables
8trainable_variables
9	variables

hlayers
imetrics
:regularization_losses
 

0
1

0
1
'2
(3
 
­
jlayer_regularization_losses
klayer_metrics
lnon_trainable_variables
=trainable_variables
>	variables

mlayers
nmetrics
?regularization_losses
 

0
1

0
1
)2
*3
 
­
olayer_regularization_losses
player_metrics
qnon_trainable_variables
Btrainable_variables
C	variables

rlayers
smetrics
Dregularization_losses
 

0
1

0
1
+2
,3
 
­
tlayer_regularization_losses
ulayer_metrics
vnon_trainable_variables
Gtrainable_variables
H	variables

wlayers
xmetrics
Iregularization_losses
 

0

0
 
­
ylayer_regularization_losses
zlayer_metrics
{non_trainable_variables
Ltrainable_variables
M	variables

|layers
}metrics
Nregularization_losses

0

0
 
°
~layer_regularization_losses
layer_metrics
non_trainable_variables
Ptrainable_variables
Q	variables
layers
metrics
Rregularization_losses

0

0
 
²
 layer_regularization_losses
layer_metrics
non_trainable_variables
Ttrainable_variables
U	variables
layers
metrics
Vregularization_losses

 0

 0
 
²
 layer_regularization_losses
layer_metrics
non_trainable_variables
Xtrainable_variables
Y	variables
layers
metrics
Zregularization_losses

!0
"1

!0
"1
 
²
 layer_regularization_losses
layer_metrics
non_trainable_variables
\trainable_variables
]	variables
layers
metrics
^regularization_losses
 
 

#0
$1
 
 
 
 

%0
&1
 
 
 
 

'0
(1
 
 
 
 

)0
*1
 
 
 
 

+0
,1
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
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
­
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_18feed_forward_sub_net_1/batch_normalization_6/moving_mean<feed_forward_sub_net_1/batch_normalization_6/moving_variance1feed_forward_sub_net_1/batch_normalization_6/beta2feed_forward_sub_net_1/batch_normalization_6/gamma%feed_forward_sub_net_1/dense_5/kernel8feed_forward_sub_net_1/batch_normalization_7/moving_mean<feed_forward_sub_net_1/batch_normalization_7/moving_variance1feed_forward_sub_net_1/batch_normalization_7/beta2feed_forward_sub_net_1/batch_normalization_7/gamma%feed_forward_sub_net_1/dense_6/kernel8feed_forward_sub_net_1/batch_normalization_8/moving_mean<feed_forward_sub_net_1/batch_normalization_8/moving_variance1feed_forward_sub_net_1/batch_normalization_8/beta2feed_forward_sub_net_1/batch_normalization_8/gamma%feed_forward_sub_net_1/dense_7/kernel8feed_forward_sub_net_1/batch_normalization_9/moving_mean<feed_forward_sub_net_1/batch_normalization_9/moving_variance1feed_forward_sub_net_1/batch_normalization_9/beta2feed_forward_sub_net_1/batch_normalization_9/gamma%feed_forward_sub_net_1/dense_8/kernel9feed_forward_sub_net_1/batch_normalization_10/moving_mean=feed_forward_sub_net_1/batch_normalization_10/moving_variance2feed_forward_sub_net_1/batch_normalization_10/beta3feed_forward_sub_net_1/batch_normalization_10/gamma%feed_forward_sub_net_1/dense_9/kernel#feed_forward_sub_net_1/dense_9/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_302603
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameFfeed_forward_sub_net_1/batch_normalization_6/gamma/Read/ReadVariableOpEfeed_forward_sub_net_1/batch_normalization_6/beta/Read/ReadVariableOpFfeed_forward_sub_net_1/batch_normalization_7/gamma/Read/ReadVariableOpEfeed_forward_sub_net_1/batch_normalization_7/beta/Read/ReadVariableOpFfeed_forward_sub_net_1/batch_normalization_8/gamma/Read/ReadVariableOpEfeed_forward_sub_net_1/batch_normalization_8/beta/Read/ReadVariableOpFfeed_forward_sub_net_1/batch_normalization_9/gamma/Read/ReadVariableOpEfeed_forward_sub_net_1/batch_normalization_9/beta/Read/ReadVariableOpGfeed_forward_sub_net_1/batch_normalization_10/gamma/Read/ReadVariableOpFfeed_forward_sub_net_1/batch_normalization_10/beta/Read/ReadVariableOp9feed_forward_sub_net_1/dense_5/kernel/Read/ReadVariableOp9feed_forward_sub_net_1/dense_6/kernel/Read/ReadVariableOp9feed_forward_sub_net_1/dense_7/kernel/Read/ReadVariableOp9feed_forward_sub_net_1/dense_8/kernel/Read/ReadVariableOp9feed_forward_sub_net_1/dense_9/kernel/Read/ReadVariableOp7feed_forward_sub_net_1/dense_9/bias/Read/ReadVariableOpLfeed_forward_sub_net_1/batch_normalization_6/moving_mean/Read/ReadVariableOpPfeed_forward_sub_net_1/batch_normalization_6/moving_variance/Read/ReadVariableOpLfeed_forward_sub_net_1/batch_normalization_7/moving_mean/Read/ReadVariableOpPfeed_forward_sub_net_1/batch_normalization_7/moving_variance/Read/ReadVariableOpLfeed_forward_sub_net_1/batch_normalization_8/moving_mean/Read/ReadVariableOpPfeed_forward_sub_net_1/batch_normalization_8/moving_variance/Read/ReadVariableOpLfeed_forward_sub_net_1/batch_normalization_9/moving_mean/Read/ReadVariableOpPfeed_forward_sub_net_1/batch_normalization_9/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_1/batch_normalization_10/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_1/batch_normalization_10/moving_variance/Read/ReadVariableOpConst*'
Tin 
2*
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
__inference__traced_save_304001
÷
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename2feed_forward_sub_net_1/batch_normalization_6/gamma1feed_forward_sub_net_1/batch_normalization_6/beta2feed_forward_sub_net_1/batch_normalization_7/gamma1feed_forward_sub_net_1/batch_normalization_7/beta2feed_forward_sub_net_1/batch_normalization_8/gamma1feed_forward_sub_net_1/batch_normalization_8/beta2feed_forward_sub_net_1/batch_normalization_9/gamma1feed_forward_sub_net_1/batch_normalization_9/beta3feed_forward_sub_net_1/batch_normalization_10/gamma2feed_forward_sub_net_1/batch_normalization_10/beta%feed_forward_sub_net_1/dense_5/kernel%feed_forward_sub_net_1/dense_6/kernel%feed_forward_sub_net_1/dense_7/kernel%feed_forward_sub_net_1/dense_8/kernel%feed_forward_sub_net_1/dense_9/kernel#feed_forward_sub_net_1/dense_9/bias8feed_forward_sub_net_1/batch_normalization_6/moving_mean<feed_forward_sub_net_1/batch_normalization_6/moving_variance8feed_forward_sub_net_1/batch_normalization_7/moving_mean<feed_forward_sub_net_1/batch_normalization_7/moving_variance8feed_forward_sub_net_1/batch_normalization_8/moving_mean<feed_forward_sub_net_1/batch_normalization_8/moving_variance8feed_forward_sub_net_1/batch_normalization_9/moving_mean<feed_forward_sub_net_1/batch_normalization_9/moving_variance9feed_forward_sub_net_1/batch_normalization_10/moving_mean=feed_forward_sub_net_1/batch_normalization_10/moving_variance*&
Tin
2*
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
"__inference__traced_restore_304089Ï
±×
Æ
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_302895
xK
=batch_normalization_6_assignmovingavg_readvariableop_resource:M
?batch_normalization_6_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_6_cast_readvariableop_resource:B
4batch_normalization_6_cast_1_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:K
=batch_normalization_7_assignmovingavg_readvariableop_resource:M
?batch_normalization_7_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_7_cast_readvariableop_resource:B
4batch_normalization_7_cast_1_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:K
=batch_normalization_8_assignmovingavg_readvariableop_resource:M
?batch_normalization_8_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_8_cast_readvariableop_resource:B
4batch_normalization_8_cast_1_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:K
=batch_normalization_9_assignmovingavg_readvariableop_resource:M
?batch_normalization_9_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_9_cast_readvariableop_resource:B
4batch_normalization_9_cast_1_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:L
>batch_normalization_10_assignmovingavg_readvariableop_resource:N
@batch_normalization_10_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_10_cast_readvariableop_resource:C
5batch_normalization_10_cast_1_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity¢&batch_normalization_10/AssignMovingAvg¢5batch_normalization_10/AssignMovingAvg/ReadVariableOp¢(batch_normalization_10/AssignMovingAvg_1¢7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_10/Cast/ReadVariableOp¢,batch_normalization_10/Cast_1/ReadVariableOp¢%batch_normalization_6/AssignMovingAvg¢4batch_normalization_6/AssignMovingAvg/ReadVariableOp¢'batch_normalization_6/AssignMovingAvg_1¢6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp¢)batch_normalization_6/Cast/ReadVariableOp¢+batch_normalization_6/Cast_1/ReadVariableOp¢%batch_normalization_7/AssignMovingAvg¢4batch_normalization_7/AssignMovingAvg/ReadVariableOp¢'batch_normalization_7/AssignMovingAvg_1¢6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp¢)batch_normalization_7/Cast/ReadVariableOp¢+batch_normalization_7/Cast_1/ReadVariableOp¢%batch_normalization_8/AssignMovingAvg¢4batch_normalization_8/AssignMovingAvg/ReadVariableOp¢'batch_normalization_8/AssignMovingAvg_1¢6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp¢)batch_normalization_8/Cast/ReadVariableOp¢+batch_normalization_8/Cast_1/ReadVariableOp¢%batch_normalization_9/AssignMovingAvg¢4batch_normalization_9/AssignMovingAvg/ReadVariableOp¢'batch_normalization_9/AssignMovingAvg_1¢6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp¢)batch_normalization_9/Cast/ReadVariableOp¢+batch_normalization_9/Cast_1/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¶
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_6/moments/mean/reduction_indicesÌ
"batch_normalization_6/moments/meanMeanx=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_6/moments/mean¾
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_6/moments/StopGradientá
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencex3batch_normalization_6/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_6/moments/SquaredDifference¾
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_6/moments/variance/reduction_indices
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_6/moments/varianceÂ
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_6/moments/SqueezeÊ
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1
+batch_normalization_6/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_6/AssignMovingAvg/decayÆ
*batch_normalization_6/AssignMovingAvg/CastCast4batch_normalization_6/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_6/AssignMovingAvg/Castæ
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOpð
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_6/AssignMovingAvg/subá
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:0.batch_normalization_6/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_6/AssignMovingAvg/mul­
%batch_normalization_6/AssignMovingAvgAssignSubVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_6/AssignMovingAvg£
-batch_normalization_6/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_6/AssignMovingAvg_1/decayÌ
,batch_normalization_6/AssignMovingAvg_1/CastCast6batch_normalization_6/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_6/AssignMovingAvg_1/Castì
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpø
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_6/AssignMovingAvg_1/subé
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:00batch_normalization_6/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_6/AssignMovingAvg_1/mul·
'batch_normalization_6/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_6/AssignMovingAvg_1Å
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_6/Cast/ReadVariableOpË
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_6/batchnorm/add/yÚ
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/add¥
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/RsqrtÖ
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/mul³
%batch_normalization_6/batchnorm/mul_1Mulx'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_6/batchnorm/mul_1Ó
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/mul_2Ô
#batch_normalization_6/batchnorm/subSub1batch_normalization_6/Cast/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/subÝ
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_6/batchnorm/add_1¥
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp®
dense_5/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/MatMul¶
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_7/moments/mean/reduction_indicesã
"batch_normalization_7/moments/meanMeandense_5/MatMul:product:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_7/moments/mean¾
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_7/moments/StopGradientø
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferencedense_5/MatMul:product:03batch_normalization_7/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_7/moments/SquaredDifference¾
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_7/moments/variance/reduction_indices
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_7/moments/varianceÂ
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_7/moments/SqueezeÊ
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1
+batch_normalization_7/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_7/AssignMovingAvg/decayÆ
*batch_normalization_7/AssignMovingAvg/CastCast4batch_normalization_7/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_7/AssignMovingAvg/Castæ
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOpð
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/subá
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:0.batch_normalization_7/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/mul­
%batch_normalization_7/AssignMovingAvgAssignSubVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_7/AssignMovingAvg£
-batch_normalization_7/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_7/AssignMovingAvg_1/decayÌ
,batch_normalization_7/AssignMovingAvg_1/CastCast6batch_normalization_7/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_7/AssignMovingAvg_1/Castì
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpø
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/subé
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:00batch_normalization_7/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/mul·
'batch_normalization_7/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_7/AssignMovingAvg_1Å
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_7/Cast/ReadVariableOpË
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_7/batchnorm/add/yÚ
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/add¥
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/RsqrtÖ
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/mulÊ
%batch_normalization_7/batchnorm/mul_1Muldense_5/MatMul:product:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_7/batchnorm/mul_1Ó
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/mul_2Ô
#batch_normalization_7/batchnorm/subSub1batch_normalization_7/Cast/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/subÝ
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_7/batchnorm/add_1q
ReluRelu)batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¥
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMulRelu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMul¶
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_8/moments/mean/reduction_indicesã
"batch_normalization_8/moments/meanMeandense_6/MatMul:product:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_8/moments/mean¾
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_8/moments/StopGradientø
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferencedense_6/MatMul:product:03batch_normalization_8/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_8/moments/SquaredDifference¾
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_8/moments/variance/reduction_indices
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_8/moments/varianceÂ
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_8/moments/SqueezeÊ
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1
+batch_normalization_8/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_8/AssignMovingAvg/decayÆ
*batch_normalization_8/AssignMovingAvg/CastCast4batch_normalization_8/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_8/AssignMovingAvg/Castæ
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOpð
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_8/AssignMovingAvg/subá
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:0.batch_normalization_8/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_8/AssignMovingAvg/mul­
%batch_normalization_8/AssignMovingAvgAssignSubVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_8/AssignMovingAvg£
-batch_normalization_8/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_8/AssignMovingAvg_1/decayÌ
,batch_normalization_8/AssignMovingAvg_1/CastCast6batch_normalization_8/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_8/AssignMovingAvg_1/Castì
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpø
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_8/AssignMovingAvg_1/subé
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:00batch_normalization_8/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_8/AssignMovingAvg_1/mul·
'batch_normalization_8/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_8/AssignMovingAvg_1Å
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_8/Cast/ReadVariableOpË
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_8/batchnorm/add/yÚ
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/add¥
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/RsqrtÖ
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/mulÊ
%batch_normalization_8/batchnorm/mul_1Muldense_6/MatMul:product:0'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_8/batchnorm/mul_1Ó
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/mul_2Ô
#batch_normalization_8/batchnorm/subSub1batch_normalization_8/Cast/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/subÝ
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_8/batchnorm/add_1u
Relu_1Relu)batch_normalization_8/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1¥
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp
dense_7/MatMulMatMulRelu_1:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMul¶
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_9/moments/mean/reduction_indicesã
"batch_normalization_9/moments/meanMeandense_7/MatMul:product:0=batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_9/moments/mean¾
*batch_normalization_9/moments/StopGradientStopGradient+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_9/moments/StopGradientø
/batch_normalization_9/moments/SquaredDifferenceSquaredDifferencedense_7/MatMul:product:03batch_normalization_9/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_9/moments/SquaredDifference¾
8batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_9/moments/variance/reduction_indices
&batch_normalization_9/moments/varianceMean3batch_normalization_9/moments/SquaredDifference:z:0Abatch_normalization_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_9/moments/varianceÂ
%batch_normalization_9/moments/SqueezeSqueeze+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_9/moments/SqueezeÊ
'batch_normalization_9/moments/Squeeze_1Squeeze/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_9/moments/Squeeze_1
+batch_normalization_9/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_9/AssignMovingAvg/decayÆ
*batch_normalization_9/AssignMovingAvg/CastCast4batch_normalization_9/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_9/AssignMovingAvg/Castæ
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_9_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOpð
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/subá
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:0.batch_normalization_9/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/mul­
%batch_normalization_9/AssignMovingAvgAssignSubVariableOp=batch_normalization_9_assignmovingavg_readvariableop_resource-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_9/AssignMovingAvg£
-batch_normalization_9/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_9/AssignMovingAvg_1/decayÌ
,batch_normalization_9/AssignMovingAvg_1/CastCast6batch_normalization_9/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_9/AssignMovingAvg_1/Castì
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_9_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpø
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/subé
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:00batch_normalization_9/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/mul·
'batch_normalization_9/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_9_assignmovingavg_1_readvariableop_resource/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_9/AssignMovingAvg_1Å
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_9/Cast/ReadVariableOpË
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_9/batchnorm/add/yÚ
#batch_normalization_9/batchnorm/addAddV20batch_normalization_9/moments/Squeeze_1:output:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/add¥
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/RsqrtÖ
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/mulÊ
%batch_normalization_9/batchnorm/mul_1Muldense_7/MatMul:product:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_9/batchnorm/mul_1Ó
%batch_normalization_9/batchnorm/mul_2Mul.batch_normalization_9/moments/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/mul_2Ô
#batch_normalization_9/batchnorm/subSub1batch_normalization_9/Cast/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/subÝ
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_9/batchnorm/add_1u
Relu_2Relu)batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¥
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulRelu_2:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/MatMul¸
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_10/moments/mean/reduction_indicesæ
#batch_normalization_10/moments/meanMeandense_8/MatMul:product:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_10/moments/meanÁ
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_10/moments/StopGradientû
0batch_normalization_10/moments/SquaredDifferenceSquaredDifferencedense_8/MatMul:product:04batch_normalization_10/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_10/moments/SquaredDifferenceÀ
9batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_10/moments/variance/reduction_indices
'batch_normalization_10/moments/varianceMean4batch_normalization_10/moments/SquaredDifference:z:0Bbatch_normalization_10/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_10/moments/varianceÅ
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_10/moments/SqueezeÍ
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_10/moments/Squeeze_1¡
,batch_normalization_10/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_10/AssignMovingAvg/decayÉ
+batch_normalization_10/AssignMovingAvg/CastCast5batch_normalization_10/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_10/AssignMovingAvg/Casté
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_10_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOpô
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/subå
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:0/batch_normalization_10/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/mul²
&batch_normalization_10/AssignMovingAvgAssignSubVariableOp>batch_normalization_10_assignmovingavg_readvariableop_resource.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_10/AssignMovingAvg¥
.batch_normalization_10/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_10/AssignMovingAvg_1/decayÏ
-batch_normalization_10/AssignMovingAvg_1/CastCast7batch_normalization_10/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_10/AssignMovingAvg_1/Castï
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_10_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/subí
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:01batch_normalization_10/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/mul¼
(batch_normalization_10/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_10_assignmovingavg_1_readvariableop_resource0batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_10/AssignMovingAvg_1È
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_10/Cast/ReadVariableOpÎ
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_10/batchnorm/add/yÞ
$batch_normalization_10/batchnorm/addAddV21batch_normalization_10/moments/Squeeze_1:output:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/add¨
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/RsqrtÚ
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/mulÍ
&batch_normalization_10/batchnorm/mul_1Muldense_8/MatMul:product:0(batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_10/batchnorm/mul_1×
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/mul_2Ø
$batch_normalization_10/batchnorm/subSub2batch_normalization_10/Cast/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/subá
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_10/batchnorm/add_1v
Relu_3Relu*batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¥
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMulRelu_3:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp¡
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/BiasAdds
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¡
NoOpNoOp'^batch_normalization_10/AssignMovingAvg6^batch_normalization_10/AssignMovingAvg/ReadVariableOp)^batch_normalization_10/AssignMovingAvg_18^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp&^batch_normalization_6/AssignMovingAvg5^batch_normalization_6/AssignMovingAvg/ReadVariableOp(^batch_normalization_6/AssignMovingAvg_17^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp&^batch_normalization_7/AssignMovingAvg5^batch_normalization_7/AssignMovingAvg/ReadVariableOp(^batch_normalization_7/AssignMovingAvg_17^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp&^batch_normalization_8/AssignMovingAvg5^batch_normalization_8/AssignMovingAvg/ReadVariableOp(^batch_normalization_8/AssignMovingAvg_17^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp&^batch_normalization_9/AssignMovingAvg5^batch_normalization_9/AssignMovingAvg/ReadVariableOp(^batch_normalization_9/AssignMovingAvg_17^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_10/AssignMovingAvg&batch_normalization_10/AssignMovingAvg2n
5batch_normalization_10/AssignMovingAvg/ReadVariableOp5batch_normalization_10/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_10/AssignMovingAvg_1(batch_normalization_10/AssignMovingAvg_12r
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2N
%batch_normalization_6/AssignMovingAvg%batch_normalization_6/AssignMovingAvg2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_6/AssignMovingAvg_1'batch_normalization_6/AssignMovingAvg_12p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2N
%batch_normalization_7/AssignMovingAvg%batch_normalization_7/AssignMovingAvg2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_7/AssignMovingAvg_1'batch_normalization_7/AssignMovingAvg_12p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2N
%batch_normalization_8/AssignMovingAvg%batch_normalization_8/AssignMovingAvg2l
4batch_normalization_8/AssignMovingAvg/ReadVariableOp4batch_normalization_8/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_8/AssignMovingAvg_1'batch_normalization_8/AssignMovingAvg_12p
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2N
%batch_normalization_9/AssignMovingAvg%batch_normalization_9/AssignMovingAvg2l
4batch_normalization_9/AssignMovingAvg/ReadVariableOp4batch_normalization_9/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_9/AssignMovingAvg_1'batch_normalization_9/AssignMovingAvg_12p
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
×
Ò
7__inference_batch_normalization_10_layer_call_fn_303825

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3018692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê+
Ò
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_303553

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Castª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
¬
C__inference_dense_5_layer_call_and_return_conditional_losses_303832

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°

Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_303435

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Úì
ã
!__inference__wrapped_model_301119
input_1W
Ifeed_forward_sub_net_1_batch_normalization_6_cast_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_6_cast_1_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_6_cast_2_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_6_cast_3_readvariableop_resource:O
=feed_forward_sub_net_1_dense_5_matmul_readvariableop_resource:W
Ifeed_forward_sub_net_1_batch_normalization_7_cast_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_7_cast_1_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_7_cast_2_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_7_cast_3_readvariableop_resource:O
=feed_forward_sub_net_1_dense_6_matmul_readvariableop_resource:W
Ifeed_forward_sub_net_1_batch_normalization_8_cast_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_8_cast_1_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_8_cast_2_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_8_cast_3_readvariableop_resource:O
=feed_forward_sub_net_1_dense_7_matmul_readvariableop_resource:W
Ifeed_forward_sub_net_1_batch_normalization_9_cast_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_9_cast_1_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_9_cast_2_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_9_cast_3_readvariableop_resource:O
=feed_forward_sub_net_1_dense_8_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_1_batch_normalization_10_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_1_batch_normalization_10_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_1_batch_normalization_10_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_1_batch_normalization_10_cast_3_readvariableop_resource:O
=feed_forward_sub_net_1_dense_9_matmul_readvariableop_resource:L
>feed_forward_sub_net_1_dense_9_biasadd_readvariableop_resource:
identity¢Afeed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOp¢Cfeed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOp¢Cfeed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOp¢Cfeed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOp¢@feed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOp¢Bfeed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOp¢Bfeed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOp¢Bfeed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOp¢@feed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOp¢Bfeed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOp¢Bfeed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOp¢Bfeed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOp¢@feed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOp¢Bfeed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOp¢Bfeed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOp¢Bfeed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOp¢@feed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOp¢Bfeed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOp¢Bfeed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOp¢Bfeed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOp¢4feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOp¢4feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOp¢4feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOp¢4feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOp¢5feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOp¢4feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOp
@feed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOpReadVariableOpIfeed_forward_sub_net_1_batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:*
dtype02B
@feed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOp
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOp
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOp
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOpÅ
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2>
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/add/y¹
:feed_forward_sub_net_1/batch_normalization_6/batchnorm/addAddV2Jfeed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOp:value:0Efeed_forward_sub_net_1/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_6/batchnorm/addê
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/RsqrtRsqrt>feed_forward_sub_net_1/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/Rsqrt²
:feed_forward_sub_net_1/batch_normalization_6/batchnorm/mulMul@feed_forward_sub_net_1/batch_normalization_6/batchnorm/Rsqrt:y:0Jfeed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_6/batchnorm/mulþ
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul_1Mulinput_1>feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul_1²
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul_2MulHfeed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOp:value:0>feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul_2²
:feed_forward_sub_net_1/batch_normalization_6/batchnorm/subSubJfeed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOp:value:0@feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_6/batchnorm/sub¹
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/add_1AddV2@feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul_1:z:0>feed_forward_sub_net_1/batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/add_1ê
4feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOpReadVariableOp=feed_forward_sub_net_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype026
4feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOp
%feed_forward_sub_net_1/dense_5/MatMulMatMul@feed_forward_sub_net_1/batch_normalization_6/batchnorm/add_1:z:0<feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%feed_forward_sub_net_1/dense_5/MatMul
@feed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOpReadVariableOpIfeed_forward_sub_net_1_batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype02B
@feed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOp
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOp
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOp
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOpÅ
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2>
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/add/y¹
:feed_forward_sub_net_1/batch_normalization_7/batchnorm/addAddV2Jfeed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOp:value:0Efeed_forward_sub_net_1/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_7/batchnorm/addê
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/RsqrtRsqrt>feed_forward_sub_net_1/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/Rsqrt²
:feed_forward_sub_net_1/batch_normalization_7/batchnorm/mulMul@feed_forward_sub_net_1/batch_normalization_7/batchnorm/Rsqrt:y:0Jfeed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul¦
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul_1Mul/feed_forward_sub_net_1/dense_5/MatMul:product:0>feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul_1²
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul_2MulHfeed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOp:value:0>feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul_2²
:feed_forward_sub_net_1/batch_normalization_7/batchnorm/subSubJfeed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOp:value:0@feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_7/batchnorm/sub¹
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/add_1AddV2@feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul_1:z:0>feed_forward_sub_net_1/batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/add_1¶
feed_forward_sub_net_1/ReluRelu@feed_forward_sub_net_1/batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
feed_forward_sub_net_1/Reluê
4feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOpReadVariableOp=feed_forward_sub_net_1_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype026
4feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOpó
%feed_forward_sub_net_1/dense_6/MatMulMatMul)feed_forward_sub_net_1/Relu:activations:0<feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%feed_forward_sub_net_1/dense_6/MatMul
@feed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOpReadVariableOpIfeed_forward_sub_net_1_batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype02B
@feed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOp
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOp
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOp
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOpÅ
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2>
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/add/y¹
:feed_forward_sub_net_1/batch_normalization_8/batchnorm/addAddV2Jfeed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOp:value:0Efeed_forward_sub_net_1/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_8/batchnorm/addê
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/RsqrtRsqrt>feed_forward_sub_net_1/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/Rsqrt²
:feed_forward_sub_net_1/batch_normalization_8/batchnorm/mulMul@feed_forward_sub_net_1/batch_normalization_8/batchnorm/Rsqrt:y:0Jfeed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul¦
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul_1Mul/feed_forward_sub_net_1/dense_6/MatMul:product:0>feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul_1²
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul_2MulHfeed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOp:value:0>feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul_2²
:feed_forward_sub_net_1/batch_normalization_8/batchnorm/subSubJfeed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOp:value:0@feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_8/batchnorm/sub¹
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/add_1AddV2@feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul_1:z:0>feed_forward_sub_net_1/batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/add_1º
feed_forward_sub_net_1/Relu_1Relu@feed_forward_sub_net_1/batch_normalization_8/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
feed_forward_sub_net_1/Relu_1ê
4feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOpReadVariableOp=feed_forward_sub_net_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype026
4feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOpõ
%feed_forward_sub_net_1/dense_7/MatMulMatMul+feed_forward_sub_net_1/Relu_1:activations:0<feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%feed_forward_sub_net_1/dense_7/MatMul
@feed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOpReadVariableOpIfeed_forward_sub_net_1_batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:*
dtype02B
@feed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOp
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOp
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_9_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOp
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_9_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOpÅ
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2>
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/add/y¹
:feed_forward_sub_net_1/batch_normalization_9/batchnorm/addAddV2Jfeed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOp:value:0Efeed_forward_sub_net_1/batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_9/batchnorm/addê
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/RsqrtRsqrt>feed_forward_sub_net_1/batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/Rsqrt²
:feed_forward_sub_net_1/batch_normalization_9/batchnorm/mulMul@feed_forward_sub_net_1/batch_normalization_9/batchnorm/Rsqrt:y:0Jfeed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul¦
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul_1Mul/feed_forward_sub_net_1/dense_7/MatMul:product:0>feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul_1²
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul_2MulHfeed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOp:value:0>feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul_2²
:feed_forward_sub_net_1/batch_normalization_9/batchnorm/subSubJfeed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOp:value:0@feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_9/batchnorm/sub¹
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/add_1AddV2@feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul_1:z:0>feed_forward_sub_net_1/batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/add_1º
feed_forward_sub_net_1/Relu_2Relu@feed_forward_sub_net_1/batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
feed_forward_sub_net_1/Relu_2ê
4feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOpReadVariableOp=feed_forward_sub_net_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype026
4feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOpõ
%feed_forward_sub_net_1/dense_8/MatMulMatMul+feed_forward_sub_net_1/Relu_2:activations:0<feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%feed_forward_sub_net_1/dense_8/MatMul
Afeed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_1_batch_normalization_10_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOp
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_1_batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOp
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_1_batch_normalization_10_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOp
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_1_batch_normalization_10_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOpÇ
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2?
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/add/y½
;feed_forward_sub_net_1/batch_normalization_10/batchnorm/addAddV2Kfeed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_1/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_1/batch_normalization_10/batchnorm/addí
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/RsqrtRsqrt?feed_forward_sub_net_1/batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/Rsqrt¶
;feed_forward_sub_net_1/batch_normalization_10/batchnorm/mulMulAfeed_forward_sub_net_1/batch_normalization_10/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul©
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul_1Mul/feed_forward_sub_net_1/dense_8/MatMul:product:0?feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul_1¶
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul_2MulIfeed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOp:value:0?feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul_2¶
;feed_forward_sub_net_1/batch_normalization_10/batchnorm/subSubKfeed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_1/batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_1/batch_normalization_10/batchnorm/sub½
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/add_1AddV2Afeed_forward_sub_net_1/batch_normalization_10/batchnorm/mul_1:z:0?feed_forward_sub_net_1/batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/add_1»
feed_forward_sub_net_1/Relu_3ReluAfeed_forward_sub_net_1/batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
feed_forward_sub_net_1/Relu_3ê
4feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOpReadVariableOp=feed_forward_sub_net_1_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype026
4feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOpõ
%feed_forward_sub_net_1/dense_9/MatMulMatMul+feed_forward_sub_net_1/Relu_3:activations:0<feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%feed_forward_sub_net_1/dense_9/MatMulé
5feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp>feed_forward_sub_net_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOpý
&feed_forward_sub_net_1/dense_9/BiasAddBiasAdd/feed_forward_sub_net_1/dense_9/MatMul:product:0=feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&feed_forward_sub_net_1/dense_9/BiasAdd
IdentityIdentity/feed_forward_sub_net_1/dense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity÷
NoOpNoOpB^feed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOpD^feed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOpD^feed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOpD^feed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOpA^feed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOpA^feed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOpA^feed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOpA^feed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOp5^feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOp5^feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOp5^feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOp5^feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOp6^feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOp5^feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : 2
Afeed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOpAfeed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOp2
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOpCfeed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOp2
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOpCfeed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOp2
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOpCfeed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOp2
@feed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOp@feed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOp2
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOp2
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOp2
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOp2
@feed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOp@feed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOp2
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOp2
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOp2
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOp2
@feed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOp@feed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOp2
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOp2
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOp2
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOp2
@feed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOp@feed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOp2
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOp2
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOp2
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOp2l
4feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOp4feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOp2l
4feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOp4feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOp2l
4feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOp4feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOp2l
4feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOp4feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOp2n
5feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOp5feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOp2l
4feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOp4feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Õ
Ñ
6__inference_batch_normalization_9_layer_call_fn_303743

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3017032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°

Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_301475

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
|
(__inference_dense_8_layer_call_fn_303881

inputs
unknown:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3020352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê+
Ò
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_301205

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Castª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï×
Ì
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_303187
input_1K
=batch_normalization_6_assignmovingavg_readvariableop_resource:M
?batch_normalization_6_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_6_cast_readvariableop_resource:B
4batch_normalization_6_cast_1_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:K
=batch_normalization_7_assignmovingavg_readvariableop_resource:M
?batch_normalization_7_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_7_cast_readvariableop_resource:B
4batch_normalization_7_cast_1_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:K
=batch_normalization_8_assignmovingavg_readvariableop_resource:M
?batch_normalization_8_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_8_cast_readvariableop_resource:B
4batch_normalization_8_cast_1_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:K
=batch_normalization_9_assignmovingavg_readvariableop_resource:M
?batch_normalization_9_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_9_cast_readvariableop_resource:B
4batch_normalization_9_cast_1_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:L
>batch_normalization_10_assignmovingavg_readvariableop_resource:N
@batch_normalization_10_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_10_cast_readvariableop_resource:C
5batch_normalization_10_cast_1_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity¢&batch_normalization_10/AssignMovingAvg¢5batch_normalization_10/AssignMovingAvg/ReadVariableOp¢(batch_normalization_10/AssignMovingAvg_1¢7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_10/Cast/ReadVariableOp¢,batch_normalization_10/Cast_1/ReadVariableOp¢%batch_normalization_6/AssignMovingAvg¢4batch_normalization_6/AssignMovingAvg/ReadVariableOp¢'batch_normalization_6/AssignMovingAvg_1¢6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp¢)batch_normalization_6/Cast/ReadVariableOp¢+batch_normalization_6/Cast_1/ReadVariableOp¢%batch_normalization_7/AssignMovingAvg¢4batch_normalization_7/AssignMovingAvg/ReadVariableOp¢'batch_normalization_7/AssignMovingAvg_1¢6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp¢)batch_normalization_7/Cast/ReadVariableOp¢+batch_normalization_7/Cast_1/ReadVariableOp¢%batch_normalization_8/AssignMovingAvg¢4batch_normalization_8/AssignMovingAvg/ReadVariableOp¢'batch_normalization_8/AssignMovingAvg_1¢6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp¢)batch_normalization_8/Cast/ReadVariableOp¢+batch_normalization_8/Cast_1/ReadVariableOp¢%batch_normalization_9/AssignMovingAvg¢4batch_normalization_9/AssignMovingAvg/ReadVariableOp¢'batch_normalization_9/AssignMovingAvg_1¢6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp¢)batch_normalization_9/Cast/ReadVariableOp¢+batch_normalization_9/Cast_1/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOp¶
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_6/moments/mean/reduction_indicesÒ
"batch_normalization_6/moments/meanMeaninput_1=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_6/moments/mean¾
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_6/moments/StopGradientç
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferenceinput_13batch_normalization_6/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_6/moments/SquaredDifference¾
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_6/moments/variance/reduction_indices
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_6/moments/varianceÂ
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_6/moments/SqueezeÊ
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1
+batch_normalization_6/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_6/AssignMovingAvg/decayÆ
*batch_normalization_6/AssignMovingAvg/CastCast4batch_normalization_6/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_6/AssignMovingAvg/Castæ
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOpð
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_6/AssignMovingAvg/subá
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:0.batch_normalization_6/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_6/AssignMovingAvg/mul­
%batch_normalization_6/AssignMovingAvgAssignSubVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_6/AssignMovingAvg£
-batch_normalization_6/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_6/AssignMovingAvg_1/decayÌ
,batch_normalization_6/AssignMovingAvg_1/CastCast6batch_normalization_6/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_6/AssignMovingAvg_1/Castì
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpø
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_6/AssignMovingAvg_1/subé
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:00batch_normalization_6/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_6/AssignMovingAvg_1/mul·
'batch_normalization_6/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_6/AssignMovingAvg_1Å
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_6/Cast/ReadVariableOpË
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_6/batchnorm/add/yÚ
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/add¥
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/RsqrtÖ
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/mul¹
%batch_normalization_6/batchnorm/mul_1Mulinput_1'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_6/batchnorm/mul_1Ó
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/mul_2Ô
#batch_normalization_6/batchnorm/subSub1batch_normalization_6/Cast/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/subÝ
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_6/batchnorm/add_1¥
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp®
dense_5/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/MatMul¶
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_7/moments/mean/reduction_indicesã
"batch_normalization_7/moments/meanMeandense_5/MatMul:product:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_7/moments/mean¾
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_7/moments/StopGradientø
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferencedense_5/MatMul:product:03batch_normalization_7/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_7/moments/SquaredDifference¾
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_7/moments/variance/reduction_indices
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_7/moments/varianceÂ
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_7/moments/SqueezeÊ
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1
+batch_normalization_7/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_7/AssignMovingAvg/decayÆ
*batch_normalization_7/AssignMovingAvg/CastCast4batch_normalization_7/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_7/AssignMovingAvg/Castæ
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOpð
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/subá
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:0.batch_normalization_7/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/mul­
%batch_normalization_7/AssignMovingAvgAssignSubVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_7/AssignMovingAvg£
-batch_normalization_7/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_7/AssignMovingAvg_1/decayÌ
,batch_normalization_7/AssignMovingAvg_1/CastCast6batch_normalization_7/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_7/AssignMovingAvg_1/Castì
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpø
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/subé
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:00batch_normalization_7/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/mul·
'batch_normalization_7/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_7/AssignMovingAvg_1Å
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_7/Cast/ReadVariableOpË
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_7/batchnorm/add/yÚ
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/add¥
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/RsqrtÖ
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/mulÊ
%batch_normalization_7/batchnorm/mul_1Muldense_5/MatMul:product:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_7/batchnorm/mul_1Ó
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/mul_2Ô
#batch_normalization_7/batchnorm/subSub1batch_normalization_7/Cast/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/subÝ
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_7/batchnorm/add_1q
ReluRelu)batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¥
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMulRelu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMul¶
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_8/moments/mean/reduction_indicesã
"batch_normalization_8/moments/meanMeandense_6/MatMul:product:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_8/moments/mean¾
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_8/moments/StopGradientø
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferencedense_6/MatMul:product:03batch_normalization_8/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_8/moments/SquaredDifference¾
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_8/moments/variance/reduction_indices
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_8/moments/varianceÂ
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_8/moments/SqueezeÊ
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1
+batch_normalization_8/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_8/AssignMovingAvg/decayÆ
*batch_normalization_8/AssignMovingAvg/CastCast4batch_normalization_8/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_8/AssignMovingAvg/Castæ
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOpð
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_8/AssignMovingAvg/subá
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:0.batch_normalization_8/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_8/AssignMovingAvg/mul­
%batch_normalization_8/AssignMovingAvgAssignSubVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_8/AssignMovingAvg£
-batch_normalization_8/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_8/AssignMovingAvg_1/decayÌ
,batch_normalization_8/AssignMovingAvg_1/CastCast6batch_normalization_8/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_8/AssignMovingAvg_1/Castì
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpø
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_8/AssignMovingAvg_1/subé
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:00batch_normalization_8/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_8/AssignMovingAvg_1/mul·
'batch_normalization_8/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_8/AssignMovingAvg_1Å
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_8/Cast/ReadVariableOpË
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_8/batchnorm/add/yÚ
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/add¥
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/RsqrtÖ
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/mulÊ
%batch_normalization_8/batchnorm/mul_1Muldense_6/MatMul:product:0'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_8/batchnorm/mul_1Ó
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/mul_2Ô
#batch_normalization_8/batchnorm/subSub1batch_normalization_8/Cast/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/subÝ
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_8/batchnorm/add_1u
Relu_1Relu)batch_normalization_8/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1¥
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp
dense_7/MatMulMatMulRelu_1:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMul¶
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_9/moments/mean/reduction_indicesã
"batch_normalization_9/moments/meanMeandense_7/MatMul:product:0=batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_9/moments/mean¾
*batch_normalization_9/moments/StopGradientStopGradient+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_9/moments/StopGradientø
/batch_normalization_9/moments/SquaredDifferenceSquaredDifferencedense_7/MatMul:product:03batch_normalization_9/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/batch_normalization_9/moments/SquaredDifference¾
8batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_9/moments/variance/reduction_indices
&batch_normalization_9/moments/varianceMean3batch_normalization_9/moments/SquaredDifference:z:0Abatch_normalization_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_9/moments/varianceÂ
%batch_normalization_9/moments/SqueezeSqueeze+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_9/moments/SqueezeÊ
'batch_normalization_9/moments/Squeeze_1Squeeze/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_9/moments/Squeeze_1
+batch_normalization_9/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_9/AssignMovingAvg/decayÆ
*batch_normalization_9/AssignMovingAvg/CastCast4batch_normalization_9/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_9/AssignMovingAvg/Castæ
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_9_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOpð
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/subá
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:0.batch_normalization_9/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/mul­
%batch_normalization_9/AssignMovingAvgAssignSubVariableOp=batch_normalization_9_assignmovingavg_readvariableop_resource-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_9/AssignMovingAvg£
-batch_normalization_9/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_9/AssignMovingAvg_1/decayÌ
,batch_normalization_9/AssignMovingAvg_1/CastCast6batch_normalization_9/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_9/AssignMovingAvg_1/Castì
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_9_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpø
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/subé
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:00batch_normalization_9/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/mul·
'batch_normalization_9/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_9_assignmovingavg_1_readvariableop_resource/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_9/AssignMovingAvg_1Å
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_9/Cast/ReadVariableOpË
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_9/batchnorm/add/yÚ
#batch_normalization_9/batchnorm/addAddV20batch_normalization_9/moments/Squeeze_1:output:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/add¥
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/RsqrtÖ
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/mulÊ
%batch_normalization_9/batchnorm/mul_1Muldense_7/MatMul:product:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_9/batchnorm/mul_1Ó
%batch_normalization_9/batchnorm/mul_2Mul.batch_normalization_9/moments/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/mul_2Ô
#batch_normalization_9/batchnorm/subSub1batch_normalization_9/Cast/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/subÝ
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_9/batchnorm/add_1u
Relu_2Relu)batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¥
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulRelu_2:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/MatMul¸
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_10/moments/mean/reduction_indicesæ
#batch_normalization_10/moments/meanMeandense_8/MatMul:product:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_10/moments/meanÁ
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_10/moments/StopGradientû
0batch_normalization_10/moments/SquaredDifferenceSquaredDifferencedense_8/MatMul:product:04batch_normalization_10/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_10/moments/SquaredDifferenceÀ
9batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_10/moments/variance/reduction_indices
'batch_normalization_10/moments/varianceMean4batch_normalization_10/moments/SquaredDifference:z:0Bbatch_normalization_10/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_10/moments/varianceÅ
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_10/moments/SqueezeÍ
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_10/moments/Squeeze_1¡
,batch_normalization_10/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_10/AssignMovingAvg/decayÉ
+batch_normalization_10/AssignMovingAvg/CastCast5batch_normalization_10/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_10/AssignMovingAvg/Casté
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_10_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOpô
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/subå
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:0/batch_normalization_10/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/mul²
&batch_normalization_10/AssignMovingAvgAssignSubVariableOp>batch_normalization_10_assignmovingavg_readvariableop_resource.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_10/AssignMovingAvg¥
.batch_normalization_10/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_10/AssignMovingAvg_1/decayÏ
-batch_normalization_10/AssignMovingAvg_1/CastCast7batch_normalization_10/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_10/AssignMovingAvg_1/Castï
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_10_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/subí
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:01batch_normalization_10/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/mul¼
(batch_normalization_10/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_10_assignmovingavg_1_readvariableop_resource0batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_10/AssignMovingAvg_1È
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_10/Cast/ReadVariableOpÎ
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_10/batchnorm/add/yÞ
$batch_normalization_10/batchnorm/addAddV21batch_normalization_10/moments/Squeeze_1:output:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/add¨
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/RsqrtÚ
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/mulÍ
&batch_normalization_10/batchnorm/mul_1Muldense_8/MatMul:product:0(batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_10/batchnorm/mul_1×
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/mul_2Ø
$batch_normalization_10/batchnorm/subSub2batch_normalization_10/Cast/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/subá
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_10/batchnorm/add_1v
Relu_3Relu*batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¥
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMulRelu_3:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp¡
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/BiasAdds
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¡
NoOpNoOp'^batch_normalization_10/AssignMovingAvg6^batch_normalization_10/AssignMovingAvg/ReadVariableOp)^batch_normalization_10/AssignMovingAvg_18^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp&^batch_normalization_6/AssignMovingAvg5^batch_normalization_6/AssignMovingAvg/ReadVariableOp(^batch_normalization_6/AssignMovingAvg_17^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp&^batch_normalization_7/AssignMovingAvg5^batch_normalization_7/AssignMovingAvg/ReadVariableOp(^batch_normalization_7/AssignMovingAvg_17^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp&^batch_normalization_8/AssignMovingAvg5^batch_normalization_8/AssignMovingAvg/ReadVariableOp(^batch_normalization_8/AssignMovingAvg_17^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp&^batch_normalization_9/AssignMovingAvg5^batch_normalization_9/AssignMovingAvg/ReadVariableOp(^batch_normalization_9/AssignMovingAvg_17^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_10/AssignMovingAvg&batch_normalization_10/AssignMovingAvg2n
5batch_normalization_10/AssignMovingAvg/ReadVariableOp5batch_normalization_10/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_10/AssignMovingAvg_1(batch_normalization_10/AssignMovingAvg_12r
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2N
%batch_normalization_6/AssignMovingAvg%batch_normalization_6/AssignMovingAvg2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_6/AssignMovingAvg_1'batch_normalization_6/AssignMovingAvg_12p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2N
%batch_normalization_7/AssignMovingAvg%batch_normalization_7/AssignMovingAvg2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_7/AssignMovingAvg_1'batch_normalization_7/AssignMovingAvg_12p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2N
%batch_normalization_8/AssignMovingAvg%batch_normalization_8/AssignMovingAvg2l
4batch_normalization_8/AssignMovingAvg/ReadVariableOp4batch_normalization_8/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_8/AssignMovingAvg_1'batch_normalization_8/AssignMovingAvg_12p
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2N
%batch_normalization_9/AssignMovingAvg%batch_normalization_9/AssignMovingAvg2l
4batch_normalization_9/AssignMovingAvg/ReadVariableOp4batch_normalization_9/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_9/AssignMovingAvg_1'batch_normalization_9/AssignMovingAvg_12p
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
°

Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_303599

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
|
(__inference_dense_7_layer_call_fn_303867

inputs
unknown:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3020142
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
¬
C__inference_dense_7_layer_call_and_return_conditional_losses_303860

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
éC
Ã
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_302066
x*
batch_normalization_6_301956:*
batch_normalization_6_301958:*
batch_normalization_6_301960:*
batch_normalization_6_301962: 
dense_5_301973:*
batch_normalization_7_301976:*
batch_normalization_7_301978:*
batch_normalization_7_301980:*
batch_normalization_7_301982: 
dense_6_301994:*
batch_normalization_8_301997:*
batch_normalization_8_301999:*
batch_normalization_8_302001:*
batch_normalization_8_302003: 
dense_7_302015:*
batch_normalization_9_302018:*
batch_normalization_9_302020:*
batch_normalization_9_302022:*
batch_normalization_9_302024: 
dense_8_302036:+
batch_normalization_10_302039:+
batch_normalization_10_302041:+
batch_normalization_10_302043:+
batch_normalization_10_302045: 
dense_9_302060:
dense_9_302062:
identity¢.batch_normalization_10/StatefulPartitionedCall¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_6_301956batch_normalization_6_301958batch_normalization_6_301960batch_normalization_6_301962*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3011432/
-batch_normalization_6/StatefulPartitionedCall­
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_5_301973*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3019722!
dense_5/StatefulPartitionedCall·
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_7_301976batch_normalization_7_301978batch_normalization_7_301980batch_normalization_7_301982*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3013092/
-batch_normalization_7/StatefulPartitionedCall~
ReluRelu6batch_normalization_7/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
dense_6/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_6_301994*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3019932!
dense_6/StatefulPartitionedCall·
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_8_301997batch_normalization_8_301999batch_normalization_8_302001batch_normalization_8_302003*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_3014752/
-batch_normalization_8/StatefulPartitionedCall
Relu_1Relu6batch_normalization_8/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1
dense_7/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_7_302015*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3020142!
dense_7/StatefulPartitionedCall·
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_9_302018batch_normalization_9_302020batch_normalization_9_302022batch_normalization_9_302024*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3016412/
-batch_normalization_9/StatefulPartitionedCall
Relu_2Relu6batch_normalization_9/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2
dense_8/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_8_302036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3020352!
dense_8/StatefulPartitionedCall¾
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_10_302039batch_normalization_10_302041batch_normalization_10_302043batch_normalization_10_302045*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_30180720
.batch_normalization_10/StatefulPartitionedCall
Relu_3Relu7batch_normalization_10/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3
dense_9/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_9_302060dense_9_302062*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3020592!
dense_9/StatefulPartitionedCall
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityé
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ë+
Ó
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_303799

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Castª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
Ñ
6__inference_batch_normalization_9_layer_call_fn_303730

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3016412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
¬
C__inference_dense_5_layer_call_and_return_conditional_losses_301972

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì

7__inference_feed_forward_sub_net_1_layer_call_fn_303301
x
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_3020662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ê+
Ò
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_303471

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Castª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê+
Ò
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_303635

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Castª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
Ñ
6__inference_batch_normalization_6_layer_call_fn_303484

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3011432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ¦
è
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_303001
input_1@
2batch_normalization_6_cast_readvariableop_resource:B
4batch_normalization_6_cast_1_readvariableop_resource:B
4batch_normalization_6_cast_2_readvariableop_resource:B
4batch_normalization_6_cast_3_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:@
2batch_normalization_7_cast_readvariableop_resource:B
4batch_normalization_7_cast_1_readvariableop_resource:B
4batch_normalization_7_cast_2_readvariableop_resource:B
4batch_normalization_7_cast_3_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:@
2batch_normalization_8_cast_readvariableop_resource:B
4batch_normalization_8_cast_1_readvariableop_resource:B
4batch_normalization_8_cast_2_readvariableop_resource:B
4batch_normalization_8_cast_3_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:@
2batch_normalization_9_cast_readvariableop_resource:B
4batch_normalization_9_cast_1_readvariableop_resource:B
4batch_normalization_9_cast_2_readvariableop_resource:B
4batch_normalization_9_cast_3_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:A
3batch_normalization_10_cast_readvariableop_resource:C
5batch_normalization_10_cast_1_readvariableop_resource:C
5batch_normalization_10_cast_2_readvariableop_resource:C
5batch_normalization_10_cast_3_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity¢*batch_normalization_10/Cast/ReadVariableOp¢,batch_normalization_10/Cast_1/ReadVariableOp¢,batch_normalization_10/Cast_2/ReadVariableOp¢,batch_normalization_10/Cast_3/ReadVariableOp¢)batch_normalization_6/Cast/ReadVariableOp¢+batch_normalization_6/Cast_1/ReadVariableOp¢+batch_normalization_6/Cast_2/ReadVariableOp¢+batch_normalization_6/Cast_3/ReadVariableOp¢)batch_normalization_7/Cast/ReadVariableOp¢+batch_normalization_7/Cast_1/ReadVariableOp¢+batch_normalization_7/Cast_2/ReadVariableOp¢+batch_normalization_7/Cast_3/ReadVariableOp¢)batch_normalization_8/Cast/ReadVariableOp¢+batch_normalization_8/Cast_1/ReadVariableOp¢+batch_normalization_8/Cast_2/ReadVariableOp¢+batch_normalization_8/Cast_3/ReadVariableOp¢)batch_normalization_9/Cast/ReadVariableOp¢+batch_normalization_9/Cast_1/ReadVariableOp¢+batch_normalization_9/Cast_2/ReadVariableOp¢+batch_normalization_9/Cast_3/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOpÅ
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_6/Cast/ReadVariableOpË
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOpË
+batch_normalization_6/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_6/Cast_2/ReadVariableOpË
+batch_normalization_6/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_6/Cast_3/ReadVariableOp
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_6/batchnorm/add/yÝ
#batch_normalization_6/batchnorm/addAddV23batch_normalization_6/Cast_1/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/add¥
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/RsqrtÖ
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/mul¹
%batch_normalization_6/batchnorm/mul_1Mulinput_1'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_6/batchnorm/mul_1Ö
%batch_normalization_6/batchnorm/mul_2Mul1batch_normalization_6/Cast/ReadVariableOp:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/mul_2Ö
#batch_normalization_6/batchnorm/subSub3batch_normalization_6/Cast_2/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/subÝ
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_6/batchnorm/add_1¥
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp®
dense_5/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/MatMulÅ
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_7/Cast/ReadVariableOpË
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOpË
+batch_normalization_7/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_2/ReadVariableOpË
+batch_normalization_7/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_3/ReadVariableOp
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_7/batchnorm/add/yÝ
#batch_normalization_7/batchnorm/addAddV23batch_normalization_7/Cast_1/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/add¥
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/RsqrtÖ
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/mulÊ
%batch_normalization_7/batchnorm/mul_1Muldense_5/MatMul:product:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_7/batchnorm/mul_1Ö
%batch_normalization_7/batchnorm/mul_2Mul1batch_normalization_7/Cast/ReadVariableOp:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/mul_2Ö
#batch_normalization_7/batchnorm/subSub3batch_normalization_7/Cast_2/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/subÝ
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_7/batchnorm/add_1q
ReluRelu)batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¥
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMulRelu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMulÅ
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_8/Cast/ReadVariableOpË
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOpË
+batch_normalization_8/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_2/ReadVariableOpË
+batch_normalization_8/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_3/ReadVariableOp
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_8/batchnorm/add/yÝ
#batch_normalization_8/batchnorm/addAddV23batch_normalization_8/Cast_1/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/add¥
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/RsqrtÖ
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/mulÊ
%batch_normalization_8/batchnorm/mul_1Muldense_6/MatMul:product:0'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_8/batchnorm/mul_1Ö
%batch_normalization_8/batchnorm/mul_2Mul1batch_normalization_8/Cast/ReadVariableOp:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/mul_2Ö
#batch_normalization_8/batchnorm/subSub3batch_normalization_8/Cast_2/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/subÝ
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_8/batchnorm/add_1u
Relu_1Relu)batch_normalization_8/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1¥
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp
dense_7/MatMulMatMulRelu_1:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMulÅ
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_9/Cast/ReadVariableOpË
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOpË
+batch_normalization_9/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_9_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_2/ReadVariableOpË
+batch_normalization_9/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_9_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_3/ReadVariableOp
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_9/batchnorm/add/yÝ
#batch_normalization_9/batchnorm/addAddV23batch_normalization_9/Cast_1/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/add¥
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/RsqrtÖ
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/mulÊ
%batch_normalization_9/batchnorm/mul_1Muldense_7/MatMul:product:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_9/batchnorm/mul_1Ö
%batch_normalization_9/batchnorm/mul_2Mul1batch_normalization_9/Cast/ReadVariableOp:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/mul_2Ö
#batch_normalization_9/batchnorm/subSub3batch_normalization_9/Cast_2/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/subÝ
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_9/batchnorm/add_1u
Relu_2Relu)batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¥
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulRelu_2:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/MatMulÈ
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_10/Cast/ReadVariableOpÎ
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOpÎ
,batch_normalization_10/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_10_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_2/ReadVariableOpÎ
,batch_normalization_10/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_10_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_3/ReadVariableOp
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_10/batchnorm/add/yá
$batch_normalization_10/batchnorm/addAddV24batch_normalization_10/Cast_1/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/add¨
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/RsqrtÚ
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/mulÍ
&batch_normalization_10/batchnorm/mul_1Muldense_8/MatMul:product:0(batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_10/batchnorm/mul_1Ú
&batch_normalization_10/batchnorm/mul_2Mul2batch_normalization_10/Cast/ReadVariableOp:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/mul_2Ú
$batch_normalization_10/batchnorm/subSub4batch_normalization_10/Cast_2/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/subá
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_10/batchnorm/add_1v
Relu_3Relu*batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¥
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMulRelu_3:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp¡
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/BiasAdds
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¡	
NoOpNoOp+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp-^batch_normalization_10/Cast_2/ReadVariableOp-^batch_normalization_10/Cast_3/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp,^batch_normalization_6/Cast_2/ReadVariableOp,^batch_normalization_6/Cast_3/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp,^batch_normalization_7/Cast_2/ReadVariableOp,^batch_normalization_7/Cast_3/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp,^batch_normalization_8/Cast_2/ReadVariableOp,^batch_normalization_8/Cast_3/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp,^batch_normalization_9/Cast_2/ReadVariableOp,^batch_normalization_9/Cast_3/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2\
,batch_normalization_10/Cast_2/ReadVariableOp,batch_normalization_10/Cast_2/ReadVariableOp2\
,batch_normalization_10/Cast_3/ReadVariableOp,batch_normalization_10/Cast_3/ReadVariableOp2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2Z
+batch_normalization_6/Cast_2/ReadVariableOp+batch_normalization_6/Cast_2/ReadVariableOp2Z
+batch_normalization_6/Cast_3/ReadVariableOp+batch_normalization_6/Cast_3/ReadVariableOp2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2Z
+batch_normalization_7/Cast_2/ReadVariableOp+batch_normalization_7/Cast_2/ReadVariableOp2Z
+batch_normalization_7/Cast_3/ReadVariableOp+batch_normalization_7/Cast_3/ReadVariableOp2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2Z
+batch_normalization_8/Cast_2/ReadVariableOp+batch_normalization_8/Cast_2/ReadVariableOp2Z
+batch_normalization_8/Cast_3/ReadVariableOp+batch_normalization_8/Cast_3/ReadVariableOp2V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2Z
+batch_normalization_9/Cast_2/ReadVariableOp+batch_normalization_9/Cast_2/ReadVariableOp2Z
+batch_normalization_9/Cast_3/ReadVariableOp+batch_normalization_9/Cast_3/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ê+
Ò
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_301371

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Castª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ

7__inference_feed_forward_sub_net_1_layer_call_fn_303244
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_3020662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ý
¬
C__inference_dense_8_layer_call_and_return_conditional_losses_303874

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ò
7__inference_batch_normalization_10_layer_call_fn_303812

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3018072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
|
(__inference_dense_5_layer_call_fn_303839

inputs
unknown:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3019722
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
¬
C__inference_dense_6_layer_call_and_return_conditional_losses_303846

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
Ñ
6__inference_batch_normalization_6_layer_call_fn_303497

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3012052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê+
Ò
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_303717

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Castª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
¬
C__inference_dense_7_layer_call_and_return_conditional_losses_302014

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï

(__inference_dense_9_layer_call_fn_303900

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3020592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â

7__inference_feed_forward_sub_net_1_layer_call_fn_303358
x
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_3022922
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
×
Ñ
6__inference_batch_normalization_7_layer_call_fn_303566

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3013092
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
|
(__inference_dense_6_layer_call_fn_303853

inputs
unknown:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3019932
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°

Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_301641

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê+
Ò
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_301537

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Castª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ßC
Ã
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_302292
x*
batch_normalization_6_302225:*
batch_normalization_6_302227:*
batch_normalization_6_302229:*
batch_normalization_6_302231: 
dense_5_302234:*
batch_normalization_7_302237:*
batch_normalization_7_302239:*
batch_normalization_7_302241:*
batch_normalization_7_302243: 
dense_6_302247:*
batch_normalization_8_302250:*
batch_normalization_8_302252:*
batch_normalization_8_302254:*
batch_normalization_8_302256: 
dense_7_302260:*
batch_normalization_9_302263:*
batch_normalization_9_302265:*
batch_normalization_9_302267:*
batch_normalization_9_302269: 
dense_8_302273:+
batch_normalization_10_302276:+
batch_normalization_10_302278:+
batch_normalization_10_302280:+
batch_normalization_10_302282: 
dense_9_302286:
dense_9_302288:
identity¢.batch_normalization_10/StatefulPartitionedCall¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢-batch_normalization_9/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_6_302225batch_normalization_6_302227batch_normalization_6_302229batch_normalization_6_302231*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3012052/
-batch_normalization_6/StatefulPartitionedCall­
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_5_302234*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3019722!
dense_5/StatefulPartitionedCallµ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_7_302237batch_normalization_7_302239batch_normalization_7_302241batch_normalization_7_302243*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3013712/
-batch_normalization_7/StatefulPartitionedCall~
ReluRelu6batch_normalization_7/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
dense_6/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_6_302247*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3019932!
dense_6/StatefulPartitionedCallµ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_8_302250batch_normalization_8_302252batch_normalization_8_302254batch_normalization_8_302256*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_3015372/
-batch_normalization_8/StatefulPartitionedCall
Relu_1Relu6batch_normalization_8/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1
dense_7/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_7_302260*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3020142!
dense_7/StatefulPartitionedCallµ
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_9_302263batch_normalization_9_302265batch_normalization_9_302267batch_normalization_9_302269*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3017032/
-batch_normalization_9/StatefulPartitionedCall
Relu_2Relu6batch_normalization_9/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2
dense_8/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_8_302273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3020352!
dense_8/StatefulPartitionedCall¼
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_10_302276batch_normalization_10_302278batch_normalization_10_302280batch_normalization_10_302282*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_30186920
.batch_normalization_10/StatefulPartitionedCall
Relu_3Relu7batch_normalization_10/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3
dense_9/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_9_302286dense_9_302288*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3020592!
dense_9/StatefulPartitionedCall
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityé
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ë+
Ó
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_301869

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Castª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_303517

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í¦
â
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_302709
x@
2batch_normalization_6_cast_readvariableop_resource:B
4batch_normalization_6_cast_1_readvariableop_resource:B
4batch_normalization_6_cast_2_readvariableop_resource:B
4batch_normalization_6_cast_3_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:@
2batch_normalization_7_cast_readvariableop_resource:B
4batch_normalization_7_cast_1_readvariableop_resource:B
4batch_normalization_7_cast_2_readvariableop_resource:B
4batch_normalization_7_cast_3_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:@
2batch_normalization_8_cast_readvariableop_resource:B
4batch_normalization_8_cast_1_readvariableop_resource:B
4batch_normalization_8_cast_2_readvariableop_resource:B
4batch_normalization_8_cast_3_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:@
2batch_normalization_9_cast_readvariableop_resource:B
4batch_normalization_9_cast_1_readvariableop_resource:B
4batch_normalization_9_cast_2_readvariableop_resource:B
4batch_normalization_9_cast_3_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:A
3batch_normalization_10_cast_readvariableop_resource:C
5batch_normalization_10_cast_1_readvariableop_resource:C
5batch_normalization_10_cast_2_readvariableop_resource:C
5batch_normalization_10_cast_3_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:
identity¢*batch_normalization_10/Cast/ReadVariableOp¢,batch_normalization_10/Cast_1/ReadVariableOp¢,batch_normalization_10/Cast_2/ReadVariableOp¢,batch_normalization_10/Cast_3/ReadVariableOp¢)batch_normalization_6/Cast/ReadVariableOp¢+batch_normalization_6/Cast_1/ReadVariableOp¢+batch_normalization_6/Cast_2/ReadVariableOp¢+batch_normalization_6/Cast_3/ReadVariableOp¢)batch_normalization_7/Cast/ReadVariableOp¢+batch_normalization_7/Cast_1/ReadVariableOp¢+batch_normalization_7/Cast_2/ReadVariableOp¢+batch_normalization_7/Cast_3/ReadVariableOp¢)batch_normalization_8/Cast/ReadVariableOp¢+batch_normalization_8/Cast_1/ReadVariableOp¢+batch_normalization_8/Cast_2/ReadVariableOp¢+batch_normalization_8/Cast_3/ReadVariableOp¢)batch_normalization_9/Cast/ReadVariableOp¢+batch_normalization_9/Cast_1/ReadVariableOp¢+batch_normalization_9/Cast_2/ReadVariableOp¢+batch_normalization_9/Cast_3/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/MatMul/ReadVariableOp¢dense_8/MatMul/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢dense_9/MatMul/ReadVariableOpÅ
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_6/Cast/ReadVariableOpË
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOpË
+batch_normalization_6/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_6/Cast_2/ReadVariableOpË
+batch_normalization_6/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_6/Cast_3/ReadVariableOp
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_6/batchnorm/add/yÝ
#batch_normalization_6/batchnorm/addAddV23batch_normalization_6/Cast_1/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/add¥
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/RsqrtÖ
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/mul³
%batch_normalization_6/batchnorm/mul_1Mulx'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_6/batchnorm/mul_1Ö
%batch_normalization_6/batchnorm/mul_2Mul1batch_normalization_6/Cast/ReadVariableOp:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/mul_2Ö
#batch_normalization_6/batchnorm/subSub3batch_normalization_6/Cast_2/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/subÝ
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_6/batchnorm/add_1¥
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp®
dense_5/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/MatMulÅ
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_7/Cast/ReadVariableOpË
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOpË
+batch_normalization_7/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_2/ReadVariableOpË
+batch_normalization_7/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_3/ReadVariableOp
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_7/batchnorm/add/yÝ
#batch_normalization_7/batchnorm/addAddV23batch_normalization_7/Cast_1/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/add¥
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/RsqrtÖ
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/mulÊ
%batch_normalization_7/batchnorm/mul_1Muldense_5/MatMul:product:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_7/batchnorm/mul_1Ö
%batch_normalization_7/batchnorm/mul_2Mul1batch_normalization_7/Cast/ReadVariableOp:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/mul_2Ö
#batch_normalization_7/batchnorm/subSub3batch_normalization_7/Cast_2/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/subÝ
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_7/batchnorm/add_1q
ReluRelu)batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¥
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMulRelu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMulÅ
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_8/Cast/ReadVariableOpË
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOpË
+batch_normalization_8/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_2/ReadVariableOpË
+batch_normalization_8/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_3/ReadVariableOp
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_8/batchnorm/add/yÝ
#batch_normalization_8/batchnorm/addAddV23batch_normalization_8/Cast_1/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/add¥
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/RsqrtÖ
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/mulÊ
%batch_normalization_8/batchnorm/mul_1Muldense_6/MatMul:product:0'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_8/batchnorm/mul_1Ö
%batch_normalization_8/batchnorm/mul_2Mul1batch_normalization_8/Cast/ReadVariableOp:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/mul_2Ö
#batch_normalization_8/batchnorm/subSub3batch_normalization_8/Cast_2/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/subÝ
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_8/batchnorm/add_1u
Relu_1Relu)batch_normalization_8/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1¥
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp
dense_7/MatMulMatMulRelu_1:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMulÅ
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_9/Cast/ReadVariableOpË
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOpË
+batch_normalization_9/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_9_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_2/ReadVariableOpË
+batch_normalization_9/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_9_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_3/ReadVariableOp
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2'
%batch_normalization_9/batchnorm/add/yÝ
#batch_normalization_9/batchnorm/addAddV23batch_normalization_9/Cast_1/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/add¥
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/RsqrtÖ
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/mulÊ
%batch_normalization_9/batchnorm/mul_1Muldense_7/MatMul:product:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_9/batchnorm/mul_1Ö
%batch_normalization_9/batchnorm/mul_2Mul1batch_normalization_9/Cast/ReadVariableOp:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/mul_2Ö
#batch_normalization_9/batchnorm/subSub3batch_normalization_9/Cast_2/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/subÝ
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batch_normalization_9/batchnorm/add_1u
Relu_2Relu)batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¥
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMulRelu_2:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/MatMulÈ
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_10/Cast/ReadVariableOpÎ
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOpÎ
,batch_normalization_10/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_10_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_2/ReadVariableOpÎ
,batch_normalization_10/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_10_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_3/ReadVariableOp
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_10/batchnorm/add/yá
$batch_normalization_10/batchnorm/addAddV24batch_normalization_10/Cast_1/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/add¨
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/RsqrtÚ
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/mulÍ
&batch_normalization_10/batchnorm/mul_1Muldense_8/MatMul:product:0(batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_10/batchnorm/mul_1Ú
&batch_normalization_10/batchnorm/mul_2Mul2batch_normalization_10/Cast/ReadVariableOp:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/mul_2Ú
$batch_normalization_10/batchnorm/subSub4batch_normalization_10/Cast_2/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/subá
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_10/batchnorm/add_1v
Relu_3Relu*batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¥
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOp
dense_9/MatMulMatMulRelu_3:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/MatMul¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp¡
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/BiasAdds
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¡	
NoOpNoOp+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp-^batch_normalization_10/Cast_2/ReadVariableOp-^batch_normalization_10/Cast_3/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp,^batch_normalization_6/Cast_2/ReadVariableOp,^batch_normalization_6/Cast_3/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp,^batch_normalization_7/Cast_2/ReadVariableOp,^batch_normalization_7/Cast_3/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp,^batch_normalization_8/Cast_2/ReadVariableOp,^batch_normalization_8/Cast_3/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp,^batch_normalization_9/Cast_2/ReadVariableOp,^batch_normalization_9/Cast_3/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2\
,batch_normalization_10/Cast_2/ReadVariableOp,batch_normalization_10/Cast_2/ReadVariableOp2\
,batch_normalization_10/Cast_3/ReadVariableOp,batch_normalization_10/Cast_3/ReadVariableOp2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2Z
+batch_normalization_6/Cast_2/ReadVariableOp+batch_normalization_6/Cast_2/ReadVariableOp2Z
+batch_normalization_6/Cast_3/ReadVariableOp+batch_normalization_6/Cast_3/ReadVariableOp2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2Z
+batch_normalization_7/Cast_2/ReadVariableOp+batch_normalization_7/Cast_2/ReadVariableOp2Z
+batch_normalization_7/Cast_3/ReadVariableOp+batch_normalization_7/Cast_3/ReadVariableOp2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2Z
+batch_normalization_8/Cast_2/ReadVariableOp+batch_normalization_8/Cast_2/ReadVariableOp2Z
+batch_normalization_8/Cast_3/ReadVariableOp+batch_normalization_8/Cast_3/ReadVariableOp2V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2Z
+batch_normalization_9/Cast_2/ReadVariableOp+batch_normalization_9/Cast_2/ReadVariableOp2Z
+batch_normalization_9/Cast_3/ReadVariableOp+batch_normalization_9/Cast_3/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
¥

ô
C__inference_dense_9_layer_call_and_return_conditional_losses_302059

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
Ñ
6__inference_batch_normalization_7_layer_call_fn_303579

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3013712
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
¬
C__inference_dense_8_layer_call_and_return_conditional_losses_302035

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
Ñ
6__inference_batch_normalization_8_layer_call_fn_303648

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_3014752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô

7__inference_feed_forward_sub_net_1_layer_call_fn_303415
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_3022922
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
°

Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_301143

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
¬
C__inference_dense_6_layer_call_and_return_conditional_losses_301993

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_301807

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê+
Ò
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_301703

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast¤
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Castª
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°

Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_303681

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥

ô
C__inference_dense_9_layer_call_and_return_conditional_losses_303891

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º

$__inference_signature_wrapper_302603
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_3011192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Õ
Ñ
6__inference_batch_normalization_8_layer_call_fn_303661

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_3015372
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°

Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_301309

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ëF
Ù
__inference__traced_save_304001
file_prefixQ
Msavev2_feed_forward_sub_net_1_batch_normalization_6_gamma_read_readvariableopP
Lsavev2_feed_forward_sub_net_1_batch_normalization_6_beta_read_readvariableopQ
Msavev2_feed_forward_sub_net_1_batch_normalization_7_gamma_read_readvariableopP
Lsavev2_feed_forward_sub_net_1_batch_normalization_7_beta_read_readvariableopQ
Msavev2_feed_forward_sub_net_1_batch_normalization_8_gamma_read_readvariableopP
Lsavev2_feed_forward_sub_net_1_batch_normalization_8_beta_read_readvariableopQ
Msavev2_feed_forward_sub_net_1_batch_normalization_9_gamma_read_readvariableopP
Lsavev2_feed_forward_sub_net_1_batch_normalization_9_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_1_batch_normalization_10_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_1_batch_normalization_10_beta_read_readvariableopD
@savev2_feed_forward_sub_net_1_dense_5_kernel_read_readvariableopD
@savev2_feed_forward_sub_net_1_dense_6_kernel_read_readvariableopD
@savev2_feed_forward_sub_net_1_dense_7_kernel_read_readvariableopD
@savev2_feed_forward_sub_net_1_dense_8_kernel_read_readvariableopD
@savev2_feed_forward_sub_net_1_dense_9_kernel_read_readvariableopB
>savev2_feed_forward_sub_net_1_dense_9_bias_read_readvariableopW
Ssavev2_feed_forward_sub_net_1_batch_normalization_6_moving_mean_read_readvariableop[
Wsavev2_feed_forward_sub_net_1_batch_normalization_6_moving_variance_read_readvariableopW
Ssavev2_feed_forward_sub_net_1_batch_normalization_7_moving_mean_read_readvariableop[
Wsavev2_feed_forward_sub_net_1_batch_normalization_7_moving_variance_read_readvariableopW
Ssavev2_feed_forward_sub_net_1_batch_normalization_8_moving_mean_read_readvariableop[
Wsavev2_feed_forward_sub_net_1_batch_normalization_8_moving_variance_read_readvariableopW
Ssavev2_feed_forward_sub_net_1_batch_normalization_9_moving_mean_read_readvariableop[
Wsavev2_feed_forward_sub_net_1_batch_normalization_9_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_1_batch_normalization_10_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_1_batch_normalization_10_moving_variance_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameá

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ó	
valueé	Bæ	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¾
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÙ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Msavev2_feed_forward_sub_net_1_batch_normalization_6_gamma_read_readvariableopLsavev2_feed_forward_sub_net_1_batch_normalization_6_beta_read_readvariableopMsavev2_feed_forward_sub_net_1_batch_normalization_7_gamma_read_readvariableopLsavev2_feed_forward_sub_net_1_batch_normalization_7_beta_read_readvariableopMsavev2_feed_forward_sub_net_1_batch_normalization_8_gamma_read_readvariableopLsavev2_feed_forward_sub_net_1_batch_normalization_8_beta_read_readvariableopMsavev2_feed_forward_sub_net_1_batch_normalization_9_gamma_read_readvariableopLsavev2_feed_forward_sub_net_1_batch_normalization_9_beta_read_readvariableopNsavev2_feed_forward_sub_net_1_batch_normalization_10_gamma_read_readvariableopMsavev2_feed_forward_sub_net_1_batch_normalization_10_beta_read_readvariableop@savev2_feed_forward_sub_net_1_dense_5_kernel_read_readvariableop@savev2_feed_forward_sub_net_1_dense_6_kernel_read_readvariableop@savev2_feed_forward_sub_net_1_dense_7_kernel_read_readvariableop@savev2_feed_forward_sub_net_1_dense_8_kernel_read_readvariableop@savev2_feed_forward_sub_net_1_dense_9_kernel_read_readvariableop>savev2_feed_forward_sub_net_1_dense_9_bias_read_readvariableopSsavev2_feed_forward_sub_net_1_batch_normalization_6_moving_mean_read_readvariableopWsavev2_feed_forward_sub_net_1_batch_normalization_6_moving_variance_read_readvariableopSsavev2_feed_forward_sub_net_1_batch_normalization_7_moving_mean_read_readvariableopWsavev2_feed_forward_sub_net_1_batch_normalization_7_moving_variance_read_readvariableopSsavev2_feed_forward_sub_net_1_batch_normalization_8_moving_mean_read_readvariableopWsavev2_feed_forward_sub_net_1_batch_normalization_8_moving_variance_read_readvariableopSsavev2_feed_forward_sub_net_1_batch_normalization_9_moving_mean_read_readvariableopWsavev2_feed_forward_sub_net_1_batch_normalization_9_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_1_batch_normalization_10_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_1_batch_normalization_10_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*É
_input_shapes·
´: ::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
±

R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_303763

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç{
¼
"__inference__traced_restore_304089
file_prefixQ
Cassignvariableop_feed_forward_sub_net_1_batch_normalization_6_gamma:R
Dassignvariableop_1_feed_forward_sub_net_1_batch_normalization_6_beta:S
Eassignvariableop_2_feed_forward_sub_net_1_batch_normalization_7_gamma:R
Dassignvariableop_3_feed_forward_sub_net_1_batch_normalization_7_beta:S
Eassignvariableop_4_feed_forward_sub_net_1_batch_normalization_8_gamma:R
Dassignvariableop_5_feed_forward_sub_net_1_batch_normalization_8_beta:S
Eassignvariableop_6_feed_forward_sub_net_1_batch_normalization_9_gamma:R
Dassignvariableop_7_feed_forward_sub_net_1_batch_normalization_9_beta:T
Fassignvariableop_8_feed_forward_sub_net_1_batch_normalization_10_gamma:S
Eassignvariableop_9_feed_forward_sub_net_1_batch_normalization_10_beta:K
9assignvariableop_10_feed_forward_sub_net_1_dense_5_kernel:K
9assignvariableop_11_feed_forward_sub_net_1_dense_6_kernel:K
9assignvariableop_12_feed_forward_sub_net_1_dense_7_kernel:K
9assignvariableop_13_feed_forward_sub_net_1_dense_8_kernel:K
9assignvariableop_14_feed_forward_sub_net_1_dense_9_kernel:E
7assignvariableop_15_feed_forward_sub_net_1_dense_9_bias:Z
Lassignvariableop_16_feed_forward_sub_net_1_batch_normalization_6_moving_mean:^
Passignvariableop_17_feed_forward_sub_net_1_batch_normalization_6_moving_variance:Z
Lassignvariableop_18_feed_forward_sub_net_1_batch_normalization_7_moving_mean:^
Passignvariableop_19_feed_forward_sub_net_1_batch_normalization_7_moving_variance:Z
Lassignvariableop_20_feed_forward_sub_net_1_batch_normalization_8_moving_mean:^
Passignvariableop_21_feed_forward_sub_net_1_batch_normalization_8_moving_variance:Z
Lassignvariableop_22_feed_forward_sub_net_1_batch_normalization_9_moving_mean:^
Passignvariableop_23_feed_forward_sub_net_1_batch_normalization_9_moving_variance:[
Massignvariableop_24_feed_forward_sub_net_1_batch_normalization_10_moving_mean:_
Qassignvariableop_25_feed_forward_sub_net_1_batch_normalization_10_moving_variance:
identity_27¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ç

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ó	
valueé	Bæ	B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÄ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices³
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityÂ
AssignVariableOpAssignVariableOpCassignvariableop_feed_forward_sub_net_1_batch_normalization_6_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1É
AssignVariableOp_1AssignVariableOpDassignvariableop_1_feed_forward_sub_net_1_batch_normalization_6_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ê
AssignVariableOp_2AssignVariableOpEassignvariableop_2_feed_forward_sub_net_1_batch_normalization_7_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3É
AssignVariableOp_3AssignVariableOpDassignvariableop_3_feed_forward_sub_net_1_batch_normalization_7_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ê
AssignVariableOp_4AssignVariableOpEassignvariableop_4_feed_forward_sub_net_1_batch_normalization_8_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5É
AssignVariableOp_5AssignVariableOpDassignvariableop_5_feed_forward_sub_net_1_batch_normalization_8_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ê
AssignVariableOp_6AssignVariableOpEassignvariableop_6_feed_forward_sub_net_1_batch_normalization_9_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7É
AssignVariableOp_7AssignVariableOpDassignvariableop_7_feed_forward_sub_net_1_batch_normalization_9_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ë
AssignVariableOp_8AssignVariableOpFassignvariableop_8_feed_forward_sub_net_1_batch_normalization_10_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ê
AssignVariableOp_9AssignVariableOpEassignvariableop_9_feed_forward_sub_net_1_batch_normalization_10_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Á
AssignVariableOp_10AssignVariableOp9assignvariableop_10_feed_forward_sub_net_1_dense_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_feed_forward_sub_net_1_dense_6_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Á
AssignVariableOp_12AssignVariableOp9assignvariableop_12_feed_forward_sub_net_1_dense_7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Á
AssignVariableOp_13AssignVariableOp9assignvariableop_13_feed_forward_sub_net_1_dense_8_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Á
AssignVariableOp_14AssignVariableOp9assignvariableop_14_feed_forward_sub_net_1_dense_9_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¿
AssignVariableOp_15AssignVariableOp7assignvariableop_15_feed_forward_sub_net_1_dense_9_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ô
AssignVariableOp_16AssignVariableOpLassignvariableop_16_feed_forward_sub_net_1_batch_normalization_6_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ø
AssignVariableOp_17AssignVariableOpPassignvariableop_17_feed_forward_sub_net_1_batch_normalization_6_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ô
AssignVariableOp_18AssignVariableOpLassignvariableop_18_feed_forward_sub_net_1_batch_normalization_7_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ø
AssignVariableOp_19AssignVariableOpPassignvariableop_19_feed_forward_sub_net_1_batch_normalization_7_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ô
AssignVariableOp_20AssignVariableOpLassignvariableop_20_feed_forward_sub_net_1_batch_normalization_8_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ø
AssignVariableOp_21AssignVariableOpPassignvariableop_21_feed_forward_sub_net_1_batch_normalization_8_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ô
AssignVariableOp_22AssignVariableOpLassignvariableop_22_feed_forward_sub_net_1_batch_normalization_9_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ø
AssignVariableOp_23AssignVariableOpPassignvariableop_23_feed_forward_sub_net_1_batch_normalization_9_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Õ
AssignVariableOp_24AssignVariableOpMassignvariableop_24_feed_forward_sub_net_1_batch_normalization_10_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ù
AssignVariableOp_25AssignVariableOpQassignvariableop_25_feed_forward_sub_net_1_batch_normalization_10_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_259
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_26f
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_27
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_25AssignVariableOp_252(
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
_user_specified_namefile_prefix"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:áÅ
ö
	bn_layers
dense_layers
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+&call_and_return_all_conditional_losses
_default_save_signature
__call__"
_tf_keras_model
J
0
	1

2
3
4
5"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper

0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15"
trackable_list_wrapper
æ
0
1
2
3
4
5
6
7
8
9
#10
$11
%12
&13
'14
(15
)16
*17
+18
,19
20
21
22
 23
!24
"25"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
-layer_regularization_losses
.layer_metrics
/non_trainable_variables
trainable_variables
	variables

0layers
1metrics
regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
ì
2axis
	gamma
beta
#moving_mean
$moving_variance
3trainable_variables
4	variables
5regularization_losses
6	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
7axis
	gamma
beta
%moving_mean
&moving_variance
8trainable_variables
9	variables
:regularization_losses
;	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
<axis
	gamma
beta
'moving_mean
(moving_variance
=trainable_variables
>	variables
?regularization_losses
@	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
Aaxis
	gamma
beta
)moving_mean
*moving_variance
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
Faxis
	gamma
beta
+moving_mean
,moving_variance
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
³

kernel
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
+ &call_and_return_all_conditional_losses
¡__call__"
_tf_keras_layer
³

kernel
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"
_tf_keras_layer
³

kernel
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
+¤&call_and_return_all_conditional_losses
¥__call__"
_tf_keras_layer
³

 kernel
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
+¦&call_and_return_all_conditional_losses
§__call__"
_tf_keras_layer
½

!kernel
"bias
\trainable_variables
]	variables
^regularization_losses
_	keras_api
+¨&call_and_return_all_conditional_losses
©__call__"
_tf_keras_layer
@:>22feed_forward_sub_net_1/batch_normalization_6/gamma
?:=21feed_forward_sub_net_1/batch_normalization_6/beta
@:>22feed_forward_sub_net_1/batch_normalization_7/gamma
?:=21feed_forward_sub_net_1/batch_normalization_7/beta
@:>22feed_forward_sub_net_1/batch_normalization_8/gamma
?:=21feed_forward_sub_net_1/batch_normalization_8/beta
@:>22feed_forward_sub_net_1/batch_normalization_9/gamma
?:=21feed_forward_sub_net_1/batch_normalization_9/beta
A:?23feed_forward_sub_net_1/batch_normalization_10/gamma
@:>22feed_forward_sub_net_1/batch_normalization_10/beta
7:52%feed_forward_sub_net_1/dense_5/kernel
7:52%feed_forward_sub_net_1/dense_6/kernel
7:52%feed_forward_sub_net_1/dense_7/kernel
7:52%feed_forward_sub_net_1/dense_8/kernel
7:52%feed_forward_sub_net_1/dense_9/kernel
1:/2#feed_forward_sub_net_1/dense_9/bias
H:F (28feed_forward_sub_net_1/batch_normalization_6/moving_mean
L:J (2<feed_forward_sub_net_1/batch_normalization_6/moving_variance
H:F (28feed_forward_sub_net_1/batch_normalization_7/moving_mean
L:J (2<feed_forward_sub_net_1/batch_normalization_7/moving_variance
H:F (28feed_forward_sub_net_1/batch_normalization_8/moving_mean
L:J (2<feed_forward_sub_net_1/batch_normalization_8/moving_variance
H:F (28feed_forward_sub_net_1/batch_normalization_9/moving_mean
L:J (2<feed_forward_sub_net_1/batch_normalization_9/moving_variance
I:G (29feed_forward_sub_net_1/batch_normalization_10/moving_mean
M:K (2=feed_forward_sub_net_1/batch_normalization_10/moving_variance
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
f
#0
$1
%2
&3
'4
(5
)6
*7
+8
,9"
trackable_list_wrapper
n
0
	1

2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
#2
$3"
trackable_list_wrapper
 "
trackable_list_wrapper
°
`layer_regularization_losses
alayer_metrics
bnon_trainable_variables
3trainable_variables
4	variables

clayers
dmetrics
5regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
%2
&3"
trackable_list_wrapper
 "
trackable_list_wrapper
°
elayer_regularization_losses
flayer_metrics
gnon_trainable_variables
8trainable_variables
9	variables

hlayers
imetrics
:regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
'2
(3"
trackable_list_wrapper
 "
trackable_list_wrapper
°
jlayer_regularization_losses
klayer_metrics
lnon_trainable_variables
=trainable_variables
>	variables

mlayers
nmetrics
?regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
)2
*3"
trackable_list_wrapper
 "
trackable_list_wrapper
°
olayer_regularization_losses
player_metrics
qnon_trainable_variables
Btrainable_variables
C	variables

rlayers
smetrics
Dregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
+2
,3"
trackable_list_wrapper
 "
trackable_list_wrapper
°
tlayer_regularization_losses
ulayer_metrics
vnon_trainable_variables
Gtrainable_variables
H	variables

wlayers
xmetrics
Iregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
°
ylayer_regularization_losses
zlayer_metrics
{non_trainable_variables
Ltrainable_variables
M	variables

|layers
}metrics
Nregularization_losses
¡__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
³
~layer_regularization_losses
layer_metrics
non_trainable_variables
Ptrainable_variables
Q	variables
layers
metrics
Rregularization_losses
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
non_trainable_variables
Ttrainable_variables
U	variables
layers
metrics
Vregularization_losses
¥__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
'
 0"
trackable_list_wrapper
'
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
non_trainable_variables
Xtrainable_variables
Y	variables
layers
metrics
Zregularization_losses
§__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
non_trainable_variables
\trainable_variables
]	variables
layers
metrics
^regularization_losses
©__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
+0
,1"
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
2þ
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_302709
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_302895
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_303001
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_303187«
¢²
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÌBÉ
!__inference__wrapped_model_301119input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
7__inference_feed_forward_sub_net_1_layer_call_fn_303244
7__inference_feed_forward_sub_net_1_layer_call_fn_303301
7__inference_feed_forward_sub_net_1_layer_call_fn_303358
7__inference_feed_forward_sub_net_1_layer_call_fn_303415«
¢²
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
$__inference_signature_wrapper_302603input_1"
²
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
annotationsª *
 
à2Ý
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_303435
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_303471´
«²§
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ª2§
6__inference_batch_normalization_6_layer_call_fn_303484
6__inference_batch_normalization_6_layer_call_fn_303497´
«²§
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_303517
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_303553´
«²§
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ª2§
6__inference_batch_normalization_7_layer_call_fn_303566
6__inference_batch_normalization_7_layer_call_fn_303579´
«²§
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_303599
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_303635´
«²§
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ª2§
6__inference_batch_normalization_8_layer_call_fn_303648
6__inference_batch_normalization_8_layer_call_fn_303661´
«²§
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_303681
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_303717´
«²§
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ª2§
6__inference_batch_normalization_9_layer_call_fn_303730
6__inference_batch_normalization_9_layer_call_fn_303743´
«²§
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_303763
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_303799´
«²§
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¬2©
7__inference_batch_normalization_10_layer_call_fn_303812
7__inference_batch_normalization_10_layer_call_fn_303825´
«²§
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
í2ê
C__inference_dense_5_layer_call_and_return_conditional_losses_303832¢
²
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
annotationsª *
 
Ò2Ï
(__inference_dense_5_layer_call_fn_303839¢
²
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
annotationsª *
 
í2ê
C__inference_dense_6_layer_call_and_return_conditional_losses_303846¢
²
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
annotationsª *
 
Ò2Ï
(__inference_dense_6_layer_call_fn_303853¢
²
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
annotationsª *
 
í2ê
C__inference_dense_7_layer_call_and_return_conditional_losses_303860¢
²
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
annotationsª *
 
Ò2Ï
(__inference_dense_7_layer_call_fn_303867¢
²
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
annotationsª *
 
í2ê
C__inference_dense_8_layer_call_and_return_conditional_losses_303874¢
²
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
annotationsª *
 
Ò2Ï
(__inference_dense_8_layer_call_fn_303881¢
²
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
annotationsª *
 
í2ê
C__inference_dense_9_layer_call_and_return_conditional_losses_303891¢
²
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
annotationsª *
 
Ò2Ï
(__inference_dense_9_layer_call_fn_303900¢
²
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
annotationsª *
 ©
!__inference__wrapped_model_301119#$%&'()* +,!"0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ¸
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_303763b+,3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_303799b+,3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_10_layer_call_fn_303812U+,3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_10_layer_call_fn_303825U+,3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ·
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_303435b#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_303471b#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_batch_normalization_6_layer_call_fn_303484U#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_batch_normalization_6_layer_call_fn_303497U#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ·
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_303517b%&3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_303553b%&3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_batch_normalization_7_layer_call_fn_303566U%&3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_batch_normalization_7_layer_call_fn_303579U%&3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ·
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_303599b'(3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_303635b'(3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_batch_normalization_8_layer_call_fn_303648U'(3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_batch_normalization_8_layer_call_fn_303661U'(3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ·
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_303681b)*3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_303717b)*3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_batch_normalization_9_layer_call_fn_303730U)*3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_batch_normalization_9_layer_call_fn_303743U)*3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¢
C__inference_dense_5_layer_call_and_return_conditional_losses_303832[/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
(__inference_dense_5_layer_call_fn_303839N/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
C__inference_dense_6_layer_call_and_return_conditional_losses_303846[/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
(__inference_dense_6_layer_call_fn_303853N/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
C__inference_dense_7_layer_call_and_return_conditional_losses_303860[/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
(__inference_dense_7_layer_call_fn_303867N/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
C__inference_dense_8_layer_call_and_return_conditional_losses_303874[ /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
(__inference_dense_8_layer_call_fn_303881N /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_9_layer_call_and_return_conditional_losses_303891\!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_9_layer_call_fn_303900O!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÉ
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_302709s#$%&'()* +,!".¢+
$¢!

xÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_302895s#$%&'()* +,!".¢+
$¢!

xÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ï
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_303001y#$%&'()* +,!"4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ï
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_303187y#$%&'()* +,!"4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 §
7__inference_feed_forward_sub_net_1_layer_call_fn_303244l#$%&'()* +,!"4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¡
7__inference_feed_forward_sub_net_1_layer_call_fn_303301f#$%&'()* +,!".¢+
$¢!

xÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¡
7__inference_feed_forward_sub_net_1_layer_call_fn_303358f#$%&'()* +,!".¢+
$¢!

xÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ§
7__inference_feed_forward_sub_net_1_layer_call_fn_303415l#$%&'()* +,!"4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ·
$__inference_signature_wrapper_302603#$%&'()* +,!";¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ