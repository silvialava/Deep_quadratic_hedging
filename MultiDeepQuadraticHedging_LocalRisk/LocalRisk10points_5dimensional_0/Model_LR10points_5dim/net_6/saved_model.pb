ýí
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
 "serve*2.6.02unknown8Ö
¾
3feed_forward_sub_net_6/batch_normalization_36/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*D
shared_name53feed_forward_sub_net_6/batch_normalization_36/gamma
·
Gfeed_forward_sub_net_6/batch_normalization_36/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_6/batch_normalization_36/gamma*
_output_shapes
:
*
dtype0
¼
2feed_forward_sub_net_6/batch_normalization_36/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*C
shared_name42feed_forward_sub_net_6/batch_normalization_36/beta
µ
Ffeed_forward_sub_net_6/batch_normalization_36/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_6/batch_normalization_36/beta*
_output_shapes
:
*
dtype0
¾
3feed_forward_sub_net_6/batch_normalization_37/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_6/batch_normalization_37/gamma
·
Gfeed_forward_sub_net_6/batch_normalization_37/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_6/batch_normalization_37/gamma*
_output_shapes
:*
dtype0
¼
2feed_forward_sub_net_6/batch_normalization_37/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_6/batch_normalization_37/beta
µ
Ffeed_forward_sub_net_6/batch_normalization_37/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_6/batch_normalization_37/beta*
_output_shapes
:*
dtype0
¾
3feed_forward_sub_net_6/batch_normalization_38/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_6/batch_normalization_38/gamma
·
Gfeed_forward_sub_net_6/batch_normalization_38/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_6/batch_normalization_38/gamma*
_output_shapes
:*
dtype0
¼
2feed_forward_sub_net_6/batch_normalization_38/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_6/batch_normalization_38/beta
µ
Ffeed_forward_sub_net_6/batch_normalization_38/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_6/batch_normalization_38/beta*
_output_shapes
:*
dtype0
¾
3feed_forward_sub_net_6/batch_normalization_39/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_6/batch_normalization_39/gamma
·
Gfeed_forward_sub_net_6/batch_normalization_39/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_6/batch_normalization_39/gamma*
_output_shapes
:*
dtype0
¼
2feed_forward_sub_net_6/batch_normalization_39/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_6/batch_normalization_39/beta
µ
Ffeed_forward_sub_net_6/batch_normalization_39/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_6/batch_normalization_39/beta*
_output_shapes
:*
dtype0
¾
3feed_forward_sub_net_6/batch_normalization_40/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_6/batch_normalization_40/gamma
·
Gfeed_forward_sub_net_6/batch_normalization_40/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_6/batch_normalization_40/gamma*
_output_shapes
:*
dtype0
¼
2feed_forward_sub_net_6/batch_normalization_40/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_6/batch_normalization_40/beta
µ
Ffeed_forward_sub_net_6/batch_normalization_40/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_6/batch_normalization_40/beta*
_output_shapes
:*
dtype0
Ê
9feed_forward_sub_net_6/batch_normalization_36/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*J
shared_name;9feed_forward_sub_net_6/batch_normalization_36/moving_mean
Ã
Mfeed_forward_sub_net_6/batch_normalization_36/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_6/batch_normalization_36/moving_mean*
_output_shapes
:
*
dtype0
Ò
=feed_forward_sub_net_6/batch_normalization_36/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*N
shared_name?=feed_forward_sub_net_6/batch_normalization_36/moving_variance
Ë
Qfeed_forward_sub_net_6/batch_normalization_36/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_6/batch_normalization_36/moving_variance*
_output_shapes
:
*
dtype0
Ê
9feed_forward_sub_net_6/batch_normalization_37/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_6/batch_normalization_37/moving_mean
Ã
Mfeed_forward_sub_net_6/batch_normalization_37/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_6/batch_normalization_37/moving_mean*
_output_shapes
:*
dtype0
Ò
=feed_forward_sub_net_6/batch_normalization_37/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_6/batch_normalization_37/moving_variance
Ë
Qfeed_forward_sub_net_6/batch_normalization_37/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_6/batch_normalization_37/moving_variance*
_output_shapes
:*
dtype0
Ê
9feed_forward_sub_net_6/batch_normalization_38/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_6/batch_normalization_38/moving_mean
Ã
Mfeed_forward_sub_net_6/batch_normalization_38/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_6/batch_normalization_38/moving_mean*
_output_shapes
:*
dtype0
Ò
=feed_forward_sub_net_6/batch_normalization_38/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_6/batch_normalization_38/moving_variance
Ë
Qfeed_forward_sub_net_6/batch_normalization_38/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_6/batch_normalization_38/moving_variance*
_output_shapes
:*
dtype0
Ê
9feed_forward_sub_net_6/batch_normalization_39/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_6/batch_normalization_39/moving_mean
Ã
Mfeed_forward_sub_net_6/batch_normalization_39/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_6/batch_normalization_39/moving_mean*
_output_shapes
:*
dtype0
Ò
=feed_forward_sub_net_6/batch_normalization_39/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_6/batch_normalization_39/moving_variance
Ë
Qfeed_forward_sub_net_6/batch_normalization_39/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_6/batch_normalization_39/moving_variance*
_output_shapes
:*
dtype0
Ê
9feed_forward_sub_net_6/batch_normalization_40/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_6/batch_normalization_40/moving_mean
Ã
Mfeed_forward_sub_net_6/batch_normalization_40/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_6/batch_normalization_40/moving_mean*
_output_shapes
:*
dtype0
Ò
=feed_forward_sub_net_6/batch_normalization_40/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_6/batch_normalization_40/moving_variance
Ë
Qfeed_forward_sub_net_6/batch_normalization_40/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_6/batch_normalization_40/moving_variance*
_output_shapes
:*
dtype0
¨
&feed_forward_sub_net_6/dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&feed_forward_sub_net_6/dense_30/kernel
¡
:feed_forward_sub_net_6/dense_30/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_6/dense_30/kernel*
_output_shapes

:
*
dtype0
¨
&feed_forward_sub_net_6/dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_6/dense_31/kernel
¡
:feed_forward_sub_net_6/dense_31/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_6/dense_31/kernel*
_output_shapes

:*
dtype0
¨
&feed_forward_sub_net_6/dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_6/dense_32/kernel
¡
:feed_forward_sub_net_6/dense_32/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_6/dense_32/kernel*
_output_shapes

:*
dtype0
¨
&feed_forward_sub_net_6/dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_6/dense_33/kernel
¡
:feed_forward_sub_net_6/dense_33/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_6/dense_33/kernel*
_output_shapes

:*
dtype0
¨
&feed_forward_sub_net_6/dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&feed_forward_sub_net_6/dense_34/kernel
¡
:feed_forward_sub_net_6/dense_34/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_6/dense_34/kernel*
_output_shapes

:
*
dtype0
 
$feed_forward_sub_net_6/dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$feed_forward_sub_net_6/dense_34/bias

8feed_forward_sub_net_6/dense_34/bias/Read/ReadVariableOpReadVariableOp$feed_forward_sub_net_6/dense_34/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
Ï:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*:
value:Bý9 Bö9

	bn_layers
dense_layers
	variables
trainable_variables
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
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
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
'10
(11
)12
*13
+14
,15
 
­
-metrics

.layers
	variables
trainable_variables
regularization_losses
/non_trainable_variables
0layer_regularization_losses
1layer_metrics
 

2axis
	gamma
beta
moving_mean
moving_variance
3	variables
4trainable_variables
5regularization_losses
6	keras_api

7axis
	gamma
beta
moving_mean
 moving_variance
8	variables
9trainable_variables
:regularization_losses
;	keras_api

<axis
	gamma
beta
!moving_mean
"moving_variance
=	variables
>trainable_variables
?regularization_losses
@	keras_api

Aaxis
	gamma
beta
#moving_mean
$moving_variance
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api

Faxis
	gamma
beta
%moving_mean
&moving_variance
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api

K	keras_api
^

'kernel
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
^

(kernel
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
^

)kernel
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
^

*kernel
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
h

+kernel
,bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
om
VARIABLE_VALUE3feed_forward_sub_net_6/batch_normalization_36/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_6/batch_normalization_36/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_6/batch_normalization_37/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_6/batch_normalization_37/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_6/batch_normalization_38/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_6/batch_normalization_38/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_6/batch_normalization_39/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_6/batch_normalization_39/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_6/batch_normalization_40/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_6/batch_normalization_40/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_6/batch_normalization_36/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_6/batch_normalization_36/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_6/batch_normalization_37/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_6/batch_normalization_37/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_6/batch_normalization_38/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_6/batch_normalization_38/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_6/batch_normalization_39/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_6/batch_normalization_39/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_6/batch_normalization_40/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_6/batch_normalization_40/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_6/dense_30/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_6/dense_31/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_6/dense_32/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_6/dense_33/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_6/dense_34/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$feed_forward_sub_net_6/dense_34/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
 
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
F
0
1
2
 3
!4
"5
#6
$7
%8
&9
 
 
 

0
1
2
3

0
1
 
­
`metrics

alayers
3	variables
4trainable_variables
5regularization_losses
bnon_trainable_variables
clayer_regularization_losses
dlayer_metrics
 

0
1
2
 3

0
1
 
­
emetrics

flayers
8	variables
9trainable_variables
:regularization_losses
gnon_trainable_variables
hlayer_regularization_losses
ilayer_metrics
 

0
1
!2
"3

0
1
 
­
jmetrics

klayers
=	variables
>trainable_variables
?regularization_losses
lnon_trainable_variables
mlayer_regularization_losses
nlayer_metrics
 

0
1
#2
$3

0
1
 
­
ometrics

players
B	variables
Ctrainable_variables
Dregularization_losses
qnon_trainable_variables
rlayer_regularization_losses
slayer_metrics
 

0
1
%2
&3

0
1
 
­
tmetrics

ulayers
G	variables
Htrainable_variables
Iregularization_losses
vnon_trainable_variables
wlayer_regularization_losses
xlayer_metrics
 

'0

'0
 
­
ymetrics

zlayers
L	variables
Mtrainable_variables
Nregularization_losses
{non_trainable_variables
|layer_regularization_losses
}layer_metrics

(0

(0
 
°
~metrics

layers
P	variables
Qtrainable_variables
Rregularization_losses
non_trainable_variables
 layer_regularization_losses
layer_metrics

)0

)0
 
²
metrics
layers
T	variables
Utrainable_variables
Vregularization_losses
non_trainable_variables
 layer_regularization_losses
layer_metrics

*0

*0
 
²
metrics
layers
X	variables
Ytrainable_variables
Zregularization_losses
non_trainable_variables
 layer_regularization_losses
layer_metrics

+0
,1

+0
,1
 
²
metrics
layers
\	variables
]trainable_variables
^regularization_losses
non_trainable_variables
 layer_regularization_losses
layer_metrics
 
 

0
1
 
 
 
 

0
 1
 
 
 
 

!0
"1
 
 
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
:ÿÿÿÿÿÿÿÿÿ
*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

Ã
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_19feed_forward_sub_net_6/batch_normalization_36/moving_mean=feed_forward_sub_net_6/batch_normalization_36/moving_variance2feed_forward_sub_net_6/batch_normalization_36/beta3feed_forward_sub_net_6/batch_normalization_36/gamma&feed_forward_sub_net_6/dense_30/kernel9feed_forward_sub_net_6/batch_normalization_37/moving_mean=feed_forward_sub_net_6/batch_normalization_37/moving_variance2feed_forward_sub_net_6/batch_normalization_37/beta3feed_forward_sub_net_6/batch_normalization_37/gamma&feed_forward_sub_net_6/dense_31/kernel9feed_forward_sub_net_6/batch_normalization_38/moving_mean=feed_forward_sub_net_6/batch_normalization_38/moving_variance2feed_forward_sub_net_6/batch_normalization_38/beta3feed_forward_sub_net_6/batch_normalization_38/gamma&feed_forward_sub_net_6/dense_32/kernel9feed_forward_sub_net_6/batch_normalization_39/moving_mean=feed_forward_sub_net_6/batch_normalization_39/moving_variance2feed_forward_sub_net_6/batch_normalization_39/beta3feed_forward_sub_net_6/batch_normalization_39/gamma&feed_forward_sub_net_6/dense_33/kernel9feed_forward_sub_net_6/batch_normalization_40/moving_mean=feed_forward_sub_net_6/batch_normalization_40/moving_variance2feed_forward_sub_net_6/batch_normalization_40/beta3feed_forward_sub_net_6/batch_normalization_40/gamma&feed_forward_sub_net_6/dense_34/kernel$feed_forward_sub_net_6/dense_34/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_322935
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameGfeed_forward_sub_net_6/batch_normalization_36/gamma/Read/ReadVariableOpFfeed_forward_sub_net_6/batch_normalization_36/beta/Read/ReadVariableOpGfeed_forward_sub_net_6/batch_normalization_37/gamma/Read/ReadVariableOpFfeed_forward_sub_net_6/batch_normalization_37/beta/Read/ReadVariableOpGfeed_forward_sub_net_6/batch_normalization_38/gamma/Read/ReadVariableOpFfeed_forward_sub_net_6/batch_normalization_38/beta/Read/ReadVariableOpGfeed_forward_sub_net_6/batch_normalization_39/gamma/Read/ReadVariableOpFfeed_forward_sub_net_6/batch_normalization_39/beta/Read/ReadVariableOpGfeed_forward_sub_net_6/batch_normalization_40/gamma/Read/ReadVariableOpFfeed_forward_sub_net_6/batch_normalization_40/beta/Read/ReadVariableOpMfeed_forward_sub_net_6/batch_normalization_36/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_6/batch_normalization_36/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_6/batch_normalization_37/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_6/batch_normalization_37/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_6/batch_normalization_38/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_6/batch_normalization_38/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_6/batch_normalization_39/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_6/batch_normalization_39/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_6/batch_normalization_40/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_6/batch_normalization_40/moving_variance/Read/ReadVariableOp:feed_forward_sub_net_6/dense_30/kernel/Read/ReadVariableOp:feed_forward_sub_net_6/dense_31/kernel/Read/ReadVariableOp:feed_forward_sub_net_6/dense_32/kernel/Read/ReadVariableOp:feed_forward_sub_net_6/dense_33/kernel/Read/ReadVariableOp:feed_forward_sub_net_6/dense_34/kernel/Read/ReadVariableOp8feed_forward_sub_net_6/dense_34/bias/Read/ReadVariableOpConst*'
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
__inference__traced_save_324333

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename3feed_forward_sub_net_6/batch_normalization_36/gamma2feed_forward_sub_net_6/batch_normalization_36/beta3feed_forward_sub_net_6/batch_normalization_37/gamma2feed_forward_sub_net_6/batch_normalization_37/beta3feed_forward_sub_net_6/batch_normalization_38/gamma2feed_forward_sub_net_6/batch_normalization_38/beta3feed_forward_sub_net_6/batch_normalization_39/gamma2feed_forward_sub_net_6/batch_normalization_39/beta3feed_forward_sub_net_6/batch_normalization_40/gamma2feed_forward_sub_net_6/batch_normalization_40/beta9feed_forward_sub_net_6/batch_normalization_36/moving_mean=feed_forward_sub_net_6/batch_normalization_36/moving_variance9feed_forward_sub_net_6/batch_normalization_37/moving_mean=feed_forward_sub_net_6/batch_normalization_37/moving_variance9feed_forward_sub_net_6/batch_normalization_38/moving_mean=feed_forward_sub_net_6/batch_normalization_38/moving_variance9feed_forward_sub_net_6/batch_normalization_39/moving_mean=feed_forward_sub_net_6/batch_normalization_39/moving_variance9feed_forward_sub_net_6/batch_normalization_40/moving_mean=feed_forward_sub_net_6/batch_normalization_40/moving_variance&feed_forward_sub_net_6/dense_30/kernel&feed_forward_sub_net_6/dense_31/kernel&feed_forward_sub_net_6/dense_32/kernel&feed_forward_sub_net_6/dense_33/kernel&feed_forward_sub_net_6/dense_34/kernel$feed_forward_sub_net_6/dense_34/bias*&
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
"__inference__traced_restore_324421à
ñ

)__inference_dense_34_layer_call_fn_324232

inputs
unknown:

	unknown_0:

identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3223912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë+
Ó
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_322035

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
­
D__inference_dense_30_layer_call_and_return_conditional_losses_322304

inputs0
matmul_readvariableop_resource:

identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
È
}
)__inference_dense_32_layer_call_fn_324199

inputs
unknown:
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_3223462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
ú
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_323227
xL
>batch_normalization_36_assignmovingavg_readvariableop_resource:
N
@batch_normalization_36_assignmovingavg_1_readvariableop_resource:
A
3batch_normalization_36_cast_readvariableop_resource:
C
5batch_normalization_36_cast_1_readvariableop_resource:
9
'dense_30_matmul_readvariableop_resource:
L
>batch_normalization_37_assignmovingavg_readvariableop_resource:N
@batch_normalization_37_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_37_cast_readvariableop_resource:C
5batch_normalization_37_cast_1_readvariableop_resource:9
'dense_31_matmul_readvariableop_resource:L
>batch_normalization_38_assignmovingavg_readvariableop_resource:N
@batch_normalization_38_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_38_cast_readvariableop_resource:C
5batch_normalization_38_cast_1_readvariableop_resource:9
'dense_32_matmul_readvariableop_resource:L
>batch_normalization_39_assignmovingavg_readvariableop_resource:N
@batch_normalization_39_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_39_cast_readvariableop_resource:C
5batch_normalization_39_cast_1_readvariableop_resource:9
'dense_33_matmul_readvariableop_resource:L
>batch_normalization_40_assignmovingavg_readvariableop_resource:N
@batch_normalization_40_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_40_cast_readvariableop_resource:C
5batch_normalization_40_cast_1_readvariableop_resource:9
'dense_34_matmul_readvariableop_resource:
6
(dense_34_biasadd_readvariableop_resource:

identity¢&batch_normalization_36/AssignMovingAvg¢5batch_normalization_36/AssignMovingAvg/ReadVariableOp¢(batch_normalization_36/AssignMovingAvg_1¢7batch_normalization_36/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_36/Cast/ReadVariableOp¢,batch_normalization_36/Cast_1/ReadVariableOp¢&batch_normalization_37/AssignMovingAvg¢5batch_normalization_37/AssignMovingAvg/ReadVariableOp¢(batch_normalization_37/AssignMovingAvg_1¢7batch_normalization_37/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_37/Cast/ReadVariableOp¢,batch_normalization_37/Cast_1/ReadVariableOp¢&batch_normalization_38/AssignMovingAvg¢5batch_normalization_38/AssignMovingAvg/ReadVariableOp¢(batch_normalization_38/AssignMovingAvg_1¢7batch_normalization_38/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_38/Cast/ReadVariableOp¢,batch_normalization_38/Cast_1/ReadVariableOp¢&batch_normalization_39/AssignMovingAvg¢5batch_normalization_39/AssignMovingAvg/ReadVariableOp¢(batch_normalization_39/AssignMovingAvg_1¢7batch_normalization_39/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_39/Cast/ReadVariableOp¢,batch_normalization_39/Cast_1/ReadVariableOp¢&batch_normalization_40/AssignMovingAvg¢5batch_normalization_40/AssignMovingAvg/ReadVariableOp¢(batch_normalization_40/AssignMovingAvg_1¢7batch_normalization_40/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_40/Cast/ReadVariableOp¢,batch_normalization_40/Cast_1/ReadVariableOp¢dense_30/MatMul/ReadVariableOp¢dense_31/MatMul/ReadVariableOp¢dense_32/MatMul/ReadVariableOp¢dense_33/MatMul/ReadVariableOp¢dense_34/BiasAdd/ReadVariableOp¢dense_34/MatMul/ReadVariableOp¸
5batch_normalization_36/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_36/moments/mean/reduction_indicesÏ
#batch_normalization_36/moments/meanMeanx>batch_normalization_36/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_36/moments/meanÁ
+batch_normalization_36/moments/StopGradientStopGradient,batch_normalization_36/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_36/moments/StopGradientä
0batch_normalization_36/moments/SquaredDifferenceSquaredDifferencex4batch_normalization_36/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
22
0batch_normalization_36/moments/SquaredDifferenceÀ
9batch_normalization_36/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_36/moments/variance/reduction_indices
'batch_normalization_36/moments/varianceMean4batch_normalization_36/moments/SquaredDifference:z:0Bbatch_normalization_36/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_36/moments/varianceÅ
&batch_normalization_36/moments/SqueezeSqueeze,batch_normalization_36/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_36/moments/SqueezeÍ
(batch_normalization_36/moments/Squeeze_1Squeeze0batch_normalization_36/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_36/moments/Squeeze_1¡
,batch_normalization_36/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_36/AssignMovingAvg/decayÉ
+batch_normalization_36/AssignMovingAvg/CastCast5batch_normalization_36/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_36/AssignMovingAvg/Casté
5batch_normalization_36/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_36_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_36/AssignMovingAvg/ReadVariableOpô
*batch_normalization_36/AssignMovingAvg/subSub=batch_normalization_36/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_36/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_36/AssignMovingAvg/subå
*batch_normalization_36/AssignMovingAvg/mulMul.batch_normalization_36/AssignMovingAvg/sub:z:0/batch_normalization_36/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_36/AssignMovingAvg/mul²
&batch_normalization_36/AssignMovingAvgAssignSubVariableOp>batch_normalization_36_assignmovingavg_readvariableop_resource.batch_normalization_36/AssignMovingAvg/mul:z:06^batch_normalization_36/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_36/AssignMovingAvg¥
.batch_normalization_36/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_36/AssignMovingAvg_1/decayÏ
-batch_normalization_36/AssignMovingAvg_1/CastCast7batch_normalization_36/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_36/AssignMovingAvg_1/Castï
7batch_normalization_36/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_36_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_36/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_36/AssignMovingAvg_1/subSub?batch_normalization_36/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_36/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_36/AssignMovingAvg_1/subí
,batch_normalization_36/AssignMovingAvg_1/mulMul0batch_normalization_36/AssignMovingAvg_1/sub:z:01batch_normalization_36/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_36/AssignMovingAvg_1/mul¼
(batch_normalization_36/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_36_assignmovingavg_1_readvariableop_resource0batch_normalization_36/AssignMovingAvg_1/mul:z:08^batch_normalization_36/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_36/AssignMovingAvg_1È
*batch_normalization_36/Cast/ReadVariableOpReadVariableOp3batch_normalization_36_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_36/Cast/ReadVariableOpÎ
,batch_normalization_36/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_36_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_36/Cast_1/ReadVariableOp
&batch_normalization_36/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_36/batchnorm/add/yÞ
$batch_normalization_36/batchnorm/addAddV21batch_normalization_36/moments/Squeeze_1:output:0/batch_normalization_36/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_36/batchnorm/add¨
&batch_normalization_36/batchnorm/RsqrtRsqrt(batch_normalization_36/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_36/batchnorm/RsqrtÚ
$batch_normalization_36/batchnorm/mulMul*batch_normalization_36/batchnorm/Rsqrt:y:04batch_normalization_36/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_36/batchnorm/mul¶
&batch_normalization_36/batchnorm/mul_1Mulx(batch_normalization_36/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_36/batchnorm/mul_1×
&batch_normalization_36/batchnorm/mul_2Mul/batch_normalization_36/moments/Squeeze:output:0(batch_normalization_36/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_36/batchnorm/mul_2Ø
$batch_normalization_36/batchnorm/subSub2batch_normalization_36/Cast/ReadVariableOp:value:0*batch_normalization_36/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_36/batchnorm/subá
&batch_normalization_36/batchnorm/add_1AddV2*batch_normalization_36/batchnorm/mul_1:z:0(batch_normalization_36/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_36/batchnorm/add_1¨
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_30/MatMul/ReadVariableOp²
dense_30/MatMulMatMul*batch_normalization_36/batchnorm/add_1:z:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_30/MatMul¸
5batch_normalization_37/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_37/moments/mean/reduction_indicesç
#batch_normalization_37/moments/meanMeandense_30/MatMul:product:0>batch_normalization_37/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_37/moments/meanÁ
+batch_normalization_37/moments/StopGradientStopGradient,batch_normalization_37/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_37/moments/StopGradientü
0batch_normalization_37/moments/SquaredDifferenceSquaredDifferencedense_30/MatMul:product:04batch_normalization_37/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_37/moments/SquaredDifferenceÀ
9batch_normalization_37/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_37/moments/variance/reduction_indices
'batch_normalization_37/moments/varianceMean4batch_normalization_37/moments/SquaredDifference:z:0Bbatch_normalization_37/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_37/moments/varianceÅ
&batch_normalization_37/moments/SqueezeSqueeze,batch_normalization_37/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_37/moments/SqueezeÍ
(batch_normalization_37/moments/Squeeze_1Squeeze0batch_normalization_37/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_37/moments/Squeeze_1¡
,batch_normalization_37/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_37/AssignMovingAvg/decayÉ
+batch_normalization_37/AssignMovingAvg/CastCast5batch_normalization_37/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_37/AssignMovingAvg/Casté
5batch_normalization_37/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_37_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_37/AssignMovingAvg/ReadVariableOpô
*batch_normalization_37/AssignMovingAvg/subSub=batch_normalization_37/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_37/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_37/AssignMovingAvg/subå
*batch_normalization_37/AssignMovingAvg/mulMul.batch_normalization_37/AssignMovingAvg/sub:z:0/batch_normalization_37/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_37/AssignMovingAvg/mul²
&batch_normalization_37/AssignMovingAvgAssignSubVariableOp>batch_normalization_37_assignmovingavg_readvariableop_resource.batch_normalization_37/AssignMovingAvg/mul:z:06^batch_normalization_37/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_37/AssignMovingAvg¥
.batch_normalization_37/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_37/AssignMovingAvg_1/decayÏ
-batch_normalization_37/AssignMovingAvg_1/CastCast7batch_normalization_37/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_37/AssignMovingAvg_1/Castï
7batch_normalization_37/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_37_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_37/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_37/AssignMovingAvg_1/subSub?batch_normalization_37/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_37/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_37/AssignMovingAvg_1/subí
,batch_normalization_37/AssignMovingAvg_1/mulMul0batch_normalization_37/AssignMovingAvg_1/sub:z:01batch_normalization_37/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_37/AssignMovingAvg_1/mul¼
(batch_normalization_37/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_37_assignmovingavg_1_readvariableop_resource0batch_normalization_37/AssignMovingAvg_1/mul:z:08^batch_normalization_37/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_37/AssignMovingAvg_1È
*batch_normalization_37/Cast/ReadVariableOpReadVariableOp3batch_normalization_37_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_37/Cast/ReadVariableOpÎ
,batch_normalization_37/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_37_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_37/Cast_1/ReadVariableOp
&batch_normalization_37/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_37/batchnorm/add/yÞ
$batch_normalization_37/batchnorm/addAddV21batch_normalization_37/moments/Squeeze_1:output:0/batch_normalization_37/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/add¨
&batch_normalization_37/batchnorm/RsqrtRsqrt(batch_normalization_37/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_37/batchnorm/RsqrtÚ
$batch_normalization_37/batchnorm/mulMul*batch_normalization_37/batchnorm/Rsqrt:y:04batch_normalization_37/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/mulÎ
&batch_normalization_37/batchnorm/mul_1Muldense_30/MatMul:product:0(batch_normalization_37/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_37/batchnorm/mul_1×
&batch_normalization_37/batchnorm/mul_2Mul/batch_normalization_37/moments/Squeeze:output:0(batch_normalization_37/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_37/batchnorm/mul_2Ø
$batch_normalization_37/batchnorm/subSub2batch_normalization_37/Cast/ReadVariableOp:value:0*batch_normalization_37/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/subá
&batch_normalization_37/batchnorm/add_1AddV2*batch_normalization_37/batchnorm/mul_1:z:0(batch_normalization_37/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_37/batchnorm/add_1r
ReluRelu*batch_normalization_37/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¨
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_31/MatMul/ReadVariableOp
dense_31/MatMulMatMulRelu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/MatMul¸
5batch_normalization_38/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_38/moments/mean/reduction_indicesç
#batch_normalization_38/moments/meanMeandense_31/MatMul:product:0>batch_normalization_38/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_38/moments/meanÁ
+batch_normalization_38/moments/StopGradientStopGradient,batch_normalization_38/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_38/moments/StopGradientü
0batch_normalization_38/moments/SquaredDifferenceSquaredDifferencedense_31/MatMul:product:04batch_normalization_38/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_38/moments/SquaredDifferenceÀ
9batch_normalization_38/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_38/moments/variance/reduction_indices
'batch_normalization_38/moments/varianceMean4batch_normalization_38/moments/SquaredDifference:z:0Bbatch_normalization_38/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_38/moments/varianceÅ
&batch_normalization_38/moments/SqueezeSqueeze,batch_normalization_38/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_38/moments/SqueezeÍ
(batch_normalization_38/moments/Squeeze_1Squeeze0batch_normalization_38/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_38/moments/Squeeze_1¡
,batch_normalization_38/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_38/AssignMovingAvg/decayÉ
+batch_normalization_38/AssignMovingAvg/CastCast5batch_normalization_38/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_38/AssignMovingAvg/Casté
5batch_normalization_38/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_38_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_38/AssignMovingAvg/ReadVariableOpô
*batch_normalization_38/AssignMovingAvg/subSub=batch_normalization_38/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_38/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_38/AssignMovingAvg/subå
*batch_normalization_38/AssignMovingAvg/mulMul.batch_normalization_38/AssignMovingAvg/sub:z:0/batch_normalization_38/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_38/AssignMovingAvg/mul²
&batch_normalization_38/AssignMovingAvgAssignSubVariableOp>batch_normalization_38_assignmovingavg_readvariableop_resource.batch_normalization_38/AssignMovingAvg/mul:z:06^batch_normalization_38/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_38/AssignMovingAvg¥
.batch_normalization_38/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_38/AssignMovingAvg_1/decayÏ
-batch_normalization_38/AssignMovingAvg_1/CastCast7batch_normalization_38/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_38/AssignMovingAvg_1/Castï
7batch_normalization_38/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_38_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_38/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_38/AssignMovingAvg_1/subSub?batch_normalization_38/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_38/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_38/AssignMovingAvg_1/subí
,batch_normalization_38/AssignMovingAvg_1/mulMul0batch_normalization_38/AssignMovingAvg_1/sub:z:01batch_normalization_38/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_38/AssignMovingAvg_1/mul¼
(batch_normalization_38/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_38_assignmovingavg_1_readvariableop_resource0batch_normalization_38/AssignMovingAvg_1/mul:z:08^batch_normalization_38/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_38/AssignMovingAvg_1È
*batch_normalization_38/Cast/ReadVariableOpReadVariableOp3batch_normalization_38_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_38/Cast/ReadVariableOpÎ
,batch_normalization_38/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_38_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_38/Cast_1/ReadVariableOp
&batch_normalization_38/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_38/batchnorm/add/yÞ
$batch_normalization_38/batchnorm/addAddV21batch_normalization_38/moments/Squeeze_1:output:0/batch_normalization_38/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/add¨
&batch_normalization_38/batchnorm/RsqrtRsqrt(batch_normalization_38/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_38/batchnorm/RsqrtÚ
$batch_normalization_38/batchnorm/mulMul*batch_normalization_38/batchnorm/Rsqrt:y:04batch_normalization_38/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/mulÎ
&batch_normalization_38/batchnorm/mul_1Muldense_31/MatMul:product:0(batch_normalization_38/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_38/batchnorm/mul_1×
&batch_normalization_38/batchnorm/mul_2Mul/batch_normalization_38/moments/Squeeze:output:0(batch_normalization_38/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_38/batchnorm/mul_2Ø
$batch_normalization_38/batchnorm/subSub2batch_normalization_38/Cast/ReadVariableOp:value:0*batch_normalization_38/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/subá
&batch_normalization_38/batchnorm/add_1AddV2*batch_normalization_38/batchnorm/mul_1:z:0(batch_normalization_38/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_38/batchnorm/add_1v
Relu_1Relu*batch_normalization_38/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1¨
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_32/MatMul/ReadVariableOp
dense_32/MatMulMatMulRelu_1:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_32/MatMul¸
5batch_normalization_39/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_39/moments/mean/reduction_indicesç
#batch_normalization_39/moments/meanMeandense_32/MatMul:product:0>batch_normalization_39/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_39/moments/meanÁ
+batch_normalization_39/moments/StopGradientStopGradient,batch_normalization_39/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_39/moments/StopGradientü
0batch_normalization_39/moments/SquaredDifferenceSquaredDifferencedense_32/MatMul:product:04batch_normalization_39/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_39/moments/SquaredDifferenceÀ
9batch_normalization_39/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_39/moments/variance/reduction_indices
'batch_normalization_39/moments/varianceMean4batch_normalization_39/moments/SquaredDifference:z:0Bbatch_normalization_39/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_39/moments/varianceÅ
&batch_normalization_39/moments/SqueezeSqueeze,batch_normalization_39/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_39/moments/SqueezeÍ
(batch_normalization_39/moments/Squeeze_1Squeeze0batch_normalization_39/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_39/moments/Squeeze_1¡
,batch_normalization_39/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_39/AssignMovingAvg/decayÉ
+batch_normalization_39/AssignMovingAvg/CastCast5batch_normalization_39/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_39/AssignMovingAvg/Casté
5batch_normalization_39/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_39_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_39/AssignMovingAvg/ReadVariableOpô
*batch_normalization_39/AssignMovingAvg/subSub=batch_normalization_39/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_39/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_39/AssignMovingAvg/subå
*batch_normalization_39/AssignMovingAvg/mulMul.batch_normalization_39/AssignMovingAvg/sub:z:0/batch_normalization_39/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_39/AssignMovingAvg/mul²
&batch_normalization_39/AssignMovingAvgAssignSubVariableOp>batch_normalization_39_assignmovingavg_readvariableop_resource.batch_normalization_39/AssignMovingAvg/mul:z:06^batch_normalization_39/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_39/AssignMovingAvg¥
.batch_normalization_39/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_39/AssignMovingAvg_1/decayÏ
-batch_normalization_39/AssignMovingAvg_1/CastCast7batch_normalization_39/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_39/AssignMovingAvg_1/Castï
7batch_normalization_39/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_39_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_39/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_39/AssignMovingAvg_1/subSub?batch_normalization_39/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_39/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_39/AssignMovingAvg_1/subí
,batch_normalization_39/AssignMovingAvg_1/mulMul0batch_normalization_39/AssignMovingAvg_1/sub:z:01batch_normalization_39/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_39/AssignMovingAvg_1/mul¼
(batch_normalization_39/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_39_assignmovingavg_1_readvariableop_resource0batch_normalization_39/AssignMovingAvg_1/mul:z:08^batch_normalization_39/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_39/AssignMovingAvg_1È
*batch_normalization_39/Cast/ReadVariableOpReadVariableOp3batch_normalization_39_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_39/Cast/ReadVariableOpÎ
,batch_normalization_39/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_39_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_39/Cast_1/ReadVariableOp
&batch_normalization_39/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_39/batchnorm/add/yÞ
$batch_normalization_39/batchnorm/addAddV21batch_normalization_39/moments/Squeeze_1:output:0/batch_normalization_39/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/add¨
&batch_normalization_39/batchnorm/RsqrtRsqrt(batch_normalization_39/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_39/batchnorm/RsqrtÚ
$batch_normalization_39/batchnorm/mulMul*batch_normalization_39/batchnorm/Rsqrt:y:04batch_normalization_39/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/mulÎ
&batch_normalization_39/batchnorm/mul_1Muldense_32/MatMul:product:0(batch_normalization_39/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_39/batchnorm/mul_1×
&batch_normalization_39/batchnorm/mul_2Mul/batch_normalization_39/moments/Squeeze:output:0(batch_normalization_39/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_39/batchnorm/mul_2Ø
$batch_normalization_39/batchnorm/subSub2batch_normalization_39/Cast/ReadVariableOp:value:0*batch_normalization_39/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/subá
&batch_normalization_39/batchnorm/add_1AddV2*batch_normalization_39/batchnorm/mul_1:z:0(batch_normalization_39/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_39/batchnorm/add_1v
Relu_2Relu*batch_normalization_39/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¨
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_33/MatMul/ReadVariableOp
dense_33/MatMulMatMulRelu_2:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_33/MatMul¸
5batch_normalization_40/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_40/moments/mean/reduction_indicesç
#batch_normalization_40/moments/meanMeandense_33/MatMul:product:0>batch_normalization_40/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_40/moments/meanÁ
+batch_normalization_40/moments/StopGradientStopGradient,batch_normalization_40/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_40/moments/StopGradientü
0batch_normalization_40/moments/SquaredDifferenceSquaredDifferencedense_33/MatMul:product:04batch_normalization_40/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_40/moments/SquaredDifferenceÀ
9batch_normalization_40/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_40/moments/variance/reduction_indices
'batch_normalization_40/moments/varianceMean4batch_normalization_40/moments/SquaredDifference:z:0Bbatch_normalization_40/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_40/moments/varianceÅ
&batch_normalization_40/moments/SqueezeSqueeze,batch_normalization_40/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_40/moments/SqueezeÍ
(batch_normalization_40/moments/Squeeze_1Squeeze0batch_normalization_40/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_40/moments/Squeeze_1¡
,batch_normalization_40/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_40/AssignMovingAvg/decayÉ
+batch_normalization_40/AssignMovingAvg/CastCast5batch_normalization_40/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_40/AssignMovingAvg/Casté
5batch_normalization_40/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_40_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_40/AssignMovingAvg/ReadVariableOpô
*batch_normalization_40/AssignMovingAvg/subSub=batch_normalization_40/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_40/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_40/AssignMovingAvg/subå
*batch_normalization_40/AssignMovingAvg/mulMul.batch_normalization_40/AssignMovingAvg/sub:z:0/batch_normalization_40/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_40/AssignMovingAvg/mul²
&batch_normalization_40/AssignMovingAvgAssignSubVariableOp>batch_normalization_40_assignmovingavg_readvariableop_resource.batch_normalization_40/AssignMovingAvg/mul:z:06^batch_normalization_40/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_40/AssignMovingAvg¥
.batch_normalization_40/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_40/AssignMovingAvg_1/decayÏ
-batch_normalization_40/AssignMovingAvg_1/CastCast7batch_normalization_40/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_40/AssignMovingAvg_1/Castï
7batch_normalization_40/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_40_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_40/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_40/AssignMovingAvg_1/subSub?batch_normalization_40/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_40/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_40/AssignMovingAvg_1/subí
,batch_normalization_40/AssignMovingAvg_1/mulMul0batch_normalization_40/AssignMovingAvg_1/sub:z:01batch_normalization_40/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_40/AssignMovingAvg_1/mul¼
(batch_normalization_40/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_40_assignmovingavg_1_readvariableop_resource0batch_normalization_40/AssignMovingAvg_1/mul:z:08^batch_normalization_40/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_40/AssignMovingAvg_1È
*batch_normalization_40/Cast/ReadVariableOpReadVariableOp3batch_normalization_40_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_40/Cast/ReadVariableOpÎ
,batch_normalization_40/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_40_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_40/Cast_1/ReadVariableOp
&batch_normalization_40/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_40/batchnorm/add/yÞ
$batch_normalization_40/batchnorm/addAddV21batch_normalization_40/moments/Squeeze_1:output:0/batch_normalization_40/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/add¨
&batch_normalization_40/batchnorm/RsqrtRsqrt(batch_normalization_40/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_40/batchnorm/RsqrtÚ
$batch_normalization_40/batchnorm/mulMul*batch_normalization_40/batchnorm/Rsqrt:y:04batch_normalization_40/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/mulÎ
&batch_normalization_40/batchnorm/mul_1Muldense_33/MatMul:product:0(batch_normalization_40/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_40/batchnorm/mul_1×
&batch_normalization_40/batchnorm/mul_2Mul/batch_normalization_40/moments/Squeeze:output:0(batch_normalization_40/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_40/batchnorm/mul_2Ø
$batch_normalization_40/batchnorm/subSub2batch_normalization_40/Cast/ReadVariableOp:value:0*batch_normalization_40/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/subá
&batch_normalization_40/batchnorm/add_1AddV2*batch_normalization_40/batchnorm/mul_1:z:0(batch_normalization_40/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_40/batchnorm/add_1v
Relu_3Relu*batch_normalization_40/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¨
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_34/MatMul/ReadVariableOp
dense_34/MatMulMatMulRelu_3:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_34/MatMul§
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_34/BiasAdd/ReadVariableOp¥
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_34/BiasAddt
IdentityIdentitydense_34/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity¿
NoOpNoOp'^batch_normalization_36/AssignMovingAvg6^batch_normalization_36/AssignMovingAvg/ReadVariableOp)^batch_normalization_36/AssignMovingAvg_18^batch_normalization_36/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_36/Cast/ReadVariableOp-^batch_normalization_36/Cast_1/ReadVariableOp'^batch_normalization_37/AssignMovingAvg6^batch_normalization_37/AssignMovingAvg/ReadVariableOp)^batch_normalization_37/AssignMovingAvg_18^batch_normalization_37/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_37/Cast/ReadVariableOp-^batch_normalization_37/Cast_1/ReadVariableOp'^batch_normalization_38/AssignMovingAvg6^batch_normalization_38/AssignMovingAvg/ReadVariableOp)^batch_normalization_38/AssignMovingAvg_18^batch_normalization_38/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_38/Cast/ReadVariableOp-^batch_normalization_38/Cast_1/ReadVariableOp'^batch_normalization_39/AssignMovingAvg6^batch_normalization_39/AssignMovingAvg/ReadVariableOp)^batch_normalization_39/AssignMovingAvg_18^batch_normalization_39/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_39/Cast/ReadVariableOp-^batch_normalization_39/Cast_1/ReadVariableOp'^batch_normalization_40/AssignMovingAvg6^batch_normalization_40/AssignMovingAvg/ReadVariableOp)^batch_normalization_40/AssignMovingAvg_18^batch_normalization_40/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_40/Cast/ReadVariableOp-^batch_normalization_40/Cast_1/ReadVariableOp^dense_30/MatMul/ReadVariableOp^dense_31/MatMul/ReadVariableOp^dense_32/MatMul/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_36/AssignMovingAvg&batch_normalization_36/AssignMovingAvg2n
5batch_normalization_36/AssignMovingAvg/ReadVariableOp5batch_normalization_36/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_36/AssignMovingAvg_1(batch_normalization_36/AssignMovingAvg_12r
7batch_normalization_36/AssignMovingAvg_1/ReadVariableOp7batch_normalization_36/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_36/Cast/ReadVariableOp*batch_normalization_36/Cast/ReadVariableOp2\
,batch_normalization_36/Cast_1/ReadVariableOp,batch_normalization_36/Cast_1/ReadVariableOp2P
&batch_normalization_37/AssignMovingAvg&batch_normalization_37/AssignMovingAvg2n
5batch_normalization_37/AssignMovingAvg/ReadVariableOp5batch_normalization_37/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_37/AssignMovingAvg_1(batch_normalization_37/AssignMovingAvg_12r
7batch_normalization_37/AssignMovingAvg_1/ReadVariableOp7batch_normalization_37/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_37/Cast/ReadVariableOp*batch_normalization_37/Cast/ReadVariableOp2\
,batch_normalization_37/Cast_1/ReadVariableOp,batch_normalization_37/Cast_1/ReadVariableOp2P
&batch_normalization_38/AssignMovingAvg&batch_normalization_38/AssignMovingAvg2n
5batch_normalization_38/AssignMovingAvg/ReadVariableOp5batch_normalization_38/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_38/AssignMovingAvg_1(batch_normalization_38/AssignMovingAvg_12r
7batch_normalization_38/AssignMovingAvg_1/ReadVariableOp7batch_normalization_38/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_38/Cast/ReadVariableOp*batch_normalization_38/Cast/ReadVariableOp2\
,batch_normalization_38/Cast_1/ReadVariableOp,batch_normalization_38/Cast_1/ReadVariableOp2P
&batch_normalization_39/AssignMovingAvg&batch_normalization_39/AssignMovingAvg2n
5batch_normalization_39/AssignMovingAvg/ReadVariableOp5batch_normalization_39/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_39/AssignMovingAvg_1(batch_normalization_39/AssignMovingAvg_12r
7batch_normalization_39/AssignMovingAvg_1/ReadVariableOp7batch_normalization_39/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_39/Cast/ReadVariableOp*batch_normalization_39/Cast/ReadVariableOp2\
,batch_normalization_39/Cast_1/ReadVariableOp,batch_normalization_39/Cast_1/ReadVariableOp2P
&batch_normalization_40/AssignMovingAvg&batch_normalization_40/AssignMovingAvg2n
5batch_normalization_40/AssignMovingAvg/ReadVariableOp5batch_normalization_40/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_40/AssignMovingAvg_1(batch_normalization_40/AssignMovingAvg_12r
7batch_normalization_40/AssignMovingAvg_1/ReadVariableOp7batch_normalization_40/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_40/Cast/ReadVariableOp*batch_normalization_40/Cast/ReadVariableOp2\
,batch_normalization_40/Cast_1/ReadVariableOp,batch_normalization_40/Cast_1/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
ÓD
â
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_322624
x+
batch_normalization_36_322557:
+
batch_normalization_36_322559:
+
batch_normalization_36_322561:
+
batch_normalization_36_322563:
!
dense_30_322566:
+
batch_normalization_37_322569:+
batch_normalization_37_322571:+
batch_normalization_37_322573:+
batch_normalization_37_322575:!
dense_31_322579:+
batch_normalization_38_322582:+
batch_normalization_38_322584:+
batch_normalization_38_322586:+
batch_normalization_38_322588:!
dense_32_322592:+
batch_normalization_39_322595:+
batch_normalization_39_322597:+
batch_normalization_39_322599:+
batch_normalization_39_322601:!
dense_33_322605:+
batch_normalization_40_322608:+
batch_normalization_40_322610:+
batch_normalization_40_322612:+
batch_normalization_40_322614:!
dense_34_322618:

dense_34_322620:

identity¢.batch_normalization_36/StatefulPartitionedCall¢.batch_normalization_37/StatefulPartitionedCall¢.batch_normalization_38/StatefulPartitionedCall¢.batch_normalization_39/StatefulPartitionedCall¢.batch_normalization_40/StatefulPartitionedCall¢ dense_30/StatefulPartitionedCall¢ dense_31/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_36_322557batch_normalization_36_322559batch_normalization_36_322561batch_normalization_36_322563*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_32153720
.batch_normalization_36/StatefulPartitionedCall²
 dense_30/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0dense_30_322566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_3223042"
 dense_30/StatefulPartitionedCall½
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0batch_normalization_37_322569batch_normalization_37_322571batch_normalization_37_322573batch_normalization_37_322575*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_32170320
.batch_normalization_37/StatefulPartitionedCall
ReluRelu7batch_normalization_37/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
 dense_31/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_31_322579*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_3223252"
 dense_31/StatefulPartitionedCall½
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0batch_normalization_38_322582batch_normalization_38_322584batch_normalization_38_322586batch_normalization_38_322588*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_32186920
.batch_normalization_38/StatefulPartitionedCall
Relu_1Relu7batch_normalization_38/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1
 dense_32/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_32_322592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_3223462"
 dense_32/StatefulPartitionedCall½
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0batch_normalization_39_322595batch_normalization_39_322597batch_normalization_39_322599batch_normalization_39_322601*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_32203520
.batch_normalization_39/StatefulPartitionedCall
Relu_2Relu7batch_normalization_39/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2
 dense_33/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_33_322605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_3223672"
 dense_33/StatefulPartitionedCall½
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0batch_normalization_40_322608batch_normalization_40_322610batch_normalization_40_322612batch_normalization_40_322614*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_32220120
.batch_normalization_40/StatefulPartitionedCall
Relu_3Relu7batch_normalization_40/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¢
 dense_34/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_34_322618dense_34_322620*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3223912"
 dense_34/StatefulPartitionedCall
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityò
NoOpNoOp/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_40/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
ÝD
â
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_322398
x+
batch_normalization_36_322288:
+
batch_normalization_36_322290:
+
batch_normalization_36_322292:
+
batch_normalization_36_322294:
!
dense_30_322305:
+
batch_normalization_37_322308:+
batch_normalization_37_322310:+
batch_normalization_37_322312:+
batch_normalization_37_322314:!
dense_31_322326:+
batch_normalization_38_322329:+
batch_normalization_38_322331:+
batch_normalization_38_322333:+
batch_normalization_38_322335:!
dense_32_322347:+
batch_normalization_39_322350:+
batch_normalization_39_322352:+
batch_normalization_39_322354:+
batch_normalization_39_322356:!
dense_33_322368:+
batch_normalization_40_322371:+
batch_normalization_40_322373:+
batch_normalization_40_322375:+
batch_normalization_40_322377:!
dense_34_322392:

dense_34_322394:

identity¢.batch_normalization_36/StatefulPartitionedCall¢.batch_normalization_37/StatefulPartitionedCall¢.batch_normalization_38/StatefulPartitionedCall¢.batch_normalization_39/StatefulPartitionedCall¢.batch_normalization_40/StatefulPartitionedCall¢ dense_30/StatefulPartitionedCall¢ dense_31/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_36_322288batch_normalization_36_322290batch_normalization_36_322292batch_normalization_36_322294*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_32147520
.batch_normalization_36/StatefulPartitionedCall²
 dense_30/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0dense_30_322305*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_3223042"
 dense_30/StatefulPartitionedCall¿
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0batch_normalization_37_322308batch_normalization_37_322310batch_normalization_37_322312batch_normalization_37_322314*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_32164120
.batch_normalization_37/StatefulPartitionedCall
ReluRelu7batch_normalization_37/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
 dense_31/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_31_322326*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_3223252"
 dense_31/StatefulPartitionedCall¿
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0batch_normalization_38_322329batch_normalization_38_322331batch_normalization_38_322333batch_normalization_38_322335*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_32180720
.batch_normalization_38/StatefulPartitionedCall
Relu_1Relu7batch_normalization_38/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1
 dense_32/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_32_322347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_32_layer_call_and_return_conditional_losses_3223462"
 dense_32/StatefulPartitionedCall¿
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0batch_normalization_39_322350batch_normalization_39_322352batch_normalization_39_322354batch_normalization_39_322356*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_32197320
.batch_normalization_39/StatefulPartitionedCall
Relu_2Relu7batch_normalization_39/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2
 dense_33/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_33_322368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_3223672"
 dense_33/StatefulPartitionedCall¿
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0batch_normalization_40_322371batch_normalization_40_322373batch_normalization_40_322375batch_normalization_40_322377*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_32213920
.batch_normalization_40/StatefulPartitionedCall
Relu_3Relu7batch_normalization_40/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¢
 dense_34/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_34_322392dense_34_322394*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_3223912"
 dense_34/StatefulPartitionedCall
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityò
NoOpNoOp/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_40/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
ë+
Ó
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_323803

inputs5
'assignmovingavg_readvariableop_resource:
7
)assignmovingavg_1_readvariableop_resource:
*
cast_readvariableop_resource:
,
cast_1_readvariableop_resource:

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

:
*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
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

:
*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
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
:
*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:
2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2
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
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:
*
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
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ë+
Ó
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_321869

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë+
Ó
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_324131

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
}
)__inference_dense_33_layer_call_fn_324213

inputs
unknown:
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_3223672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_324013

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë+
Ó
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_324049

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

õ
D__inference_dense_34_layer_call_and_return_conditional_losses_324223

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì

7__inference_feed_forward_sub_net_6_layer_call_fn_323633
x
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:


unknown_24:

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
:ÿÿÿÿÿÿÿÿÿ
*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_3223982
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
±

R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_322139

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_323931

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ò
7__inference_batch_normalization_38_layer_call_fn_323980

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_3218072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
­
D__inference_dense_30_layer_call_and_return_conditional_losses_324164

inputs0
matmul_readvariableop_resource:

identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
÷E
ï
__inference__traced_save_324333
file_prefixR
Nsavev2_feed_forward_sub_net_6_batch_normalization_36_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_6_batch_normalization_36_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_6_batch_normalization_37_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_6_batch_normalization_37_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_6_batch_normalization_38_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_6_batch_normalization_38_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_6_batch_normalization_39_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_6_batch_normalization_39_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_6_batch_normalization_40_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_6_batch_normalization_40_beta_read_readvariableopX
Tsavev2_feed_forward_sub_net_6_batch_normalization_36_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_6_batch_normalization_36_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_6_batch_normalization_37_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_6_batch_normalization_37_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_6_batch_normalization_38_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_6_batch_normalization_38_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_6_batch_normalization_39_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_6_batch_normalization_39_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_6_batch_normalization_40_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_6_batch_normalization_40_moving_variance_read_readvariableopE
Asavev2_feed_forward_sub_net_6_dense_30_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_6_dense_31_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_6_dense_32_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_6_dense_33_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_6_dense_34_kernel_read_readvariableopC
?savev2_feed_forward_sub_net_6_dense_34_bias_read_readvariableop
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
ShardedFilenameÁ	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ó
valueÉBÆB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¾
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesï
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Nsavev2_feed_forward_sub_net_6_batch_normalization_36_gamma_read_readvariableopMsavev2_feed_forward_sub_net_6_batch_normalization_36_beta_read_readvariableopNsavev2_feed_forward_sub_net_6_batch_normalization_37_gamma_read_readvariableopMsavev2_feed_forward_sub_net_6_batch_normalization_37_beta_read_readvariableopNsavev2_feed_forward_sub_net_6_batch_normalization_38_gamma_read_readvariableopMsavev2_feed_forward_sub_net_6_batch_normalization_38_beta_read_readvariableopNsavev2_feed_forward_sub_net_6_batch_normalization_39_gamma_read_readvariableopMsavev2_feed_forward_sub_net_6_batch_normalization_39_beta_read_readvariableopNsavev2_feed_forward_sub_net_6_batch_normalization_40_gamma_read_readvariableopMsavev2_feed_forward_sub_net_6_batch_normalization_40_beta_read_readvariableopTsavev2_feed_forward_sub_net_6_batch_normalization_36_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_6_batch_normalization_36_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_6_batch_normalization_37_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_6_batch_normalization_37_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_6_batch_normalization_38_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_6_batch_normalization_38_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_6_batch_normalization_39_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_6_batch_normalization_39_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_6_batch_normalization_40_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_6_batch_normalization_40_moving_variance_read_readvariableopAsavev2_feed_forward_sub_net_6_dense_30_kernel_read_readvariableopAsavev2_feed_forward_sub_net_6_dense_31_kernel_read_readvariableopAsavev2_feed_forward_sub_net_6_dense_32_kernel_read_readvariableopAsavev2_feed_forward_sub_net_6_dense_33_kernel_read_readvariableopAsavev2_feed_forward_sub_net_6_dense_34_kernel_read_readvariableop?savev2_feed_forward_sub_net_6_dense_34_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
´: :
:
:::::::::
:
:::::::::
::::
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:
: 

_output_shapes
:
: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:
:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
: 

_output_shapes
:
:

_output_shapes
: 
Ù
Ò
7__inference_batch_normalization_40_layer_call_fn_324144

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_3221392
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï
 
!__inference__wrapped_model_321451
input_1X
Jfeed_forward_sub_net_6_batch_normalization_36_cast_readvariableop_resource:
Z
Lfeed_forward_sub_net_6_batch_normalization_36_cast_1_readvariableop_resource:
Z
Lfeed_forward_sub_net_6_batch_normalization_36_cast_2_readvariableop_resource:
Z
Lfeed_forward_sub_net_6_batch_normalization_36_cast_3_readvariableop_resource:
P
>feed_forward_sub_net_6_dense_30_matmul_readvariableop_resource:
X
Jfeed_forward_sub_net_6_batch_normalization_37_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_6_batch_normalization_37_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_6_batch_normalization_37_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_6_batch_normalization_37_cast_3_readvariableop_resource:P
>feed_forward_sub_net_6_dense_31_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_6_batch_normalization_38_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_6_batch_normalization_38_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_6_batch_normalization_38_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_6_batch_normalization_38_cast_3_readvariableop_resource:P
>feed_forward_sub_net_6_dense_32_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_6_batch_normalization_39_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_6_batch_normalization_39_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_6_batch_normalization_39_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_6_batch_normalization_39_cast_3_readvariableop_resource:P
>feed_forward_sub_net_6_dense_33_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_6_batch_normalization_40_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_6_batch_normalization_40_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_6_batch_normalization_40_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_6_batch_normalization_40_cast_3_readvariableop_resource:P
>feed_forward_sub_net_6_dense_34_matmul_readvariableop_resource:
M
?feed_forward_sub_net_6_dense_34_biasadd_readvariableop_resource:

identity¢Afeed_forward_sub_net_6/batch_normalization_36/Cast/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_36/Cast_1/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_36/Cast_2/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_36/Cast_3/ReadVariableOp¢Afeed_forward_sub_net_6/batch_normalization_37/Cast/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_37/Cast_1/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_37/Cast_2/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_37/Cast_3/ReadVariableOp¢Afeed_forward_sub_net_6/batch_normalization_38/Cast/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_38/Cast_1/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_38/Cast_2/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_38/Cast_3/ReadVariableOp¢Afeed_forward_sub_net_6/batch_normalization_39/Cast/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_39/Cast_1/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_39/Cast_2/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_39/Cast_3/ReadVariableOp¢Afeed_forward_sub_net_6/batch_normalization_40/Cast/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_40/Cast_1/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_40/Cast_2/ReadVariableOp¢Cfeed_forward_sub_net_6/batch_normalization_40/Cast_3/ReadVariableOp¢5feed_forward_sub_net_6/dense_30/MatMul/ReadVariableOp¢5feed_forward_sub_net_6/dense_31/MatMul/ReadVariableOp¢5feed_forward_sub_net_6/dense_32/MatMul/ReadVariableOp¢5feed_forward_sub_net_6/dense_33/MatMul/ReadVariableOp¢6feed_forward_sub_net_6/dense_34/BiasAdd/ReadVariableOp¢5feed_forward_sub_net_6/dense_34/MatMul/ReadVariableOp
Afeed_forward_sub_net_6/batch_normalization_36/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_6_batch_normalization_36_cast_readvariableop_resource*
_output_shapes
:
*
dtype02C
Afeed_forward_sub_net_6/batch_normalization_36/Cast/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_36/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_36_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_36/Cast_1/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_36/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_36_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_36/Cast_2/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_36/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_36_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_36/Cast_3/ReadVariableOpÇ
=feed_forward_sub_net_6/batch_normalization_36/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2?
=feed_forward_sub_net_6/batch_normalization_36/batchnorm/add/y½
;feed_forward_sub_net_6/batch_normalization_36/batchnorm/addAddV2Kfeed_forward_sub_net_6/batch_normalization_36/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_6/batch_normalization_36/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_6/batch_normalization_36/batchnorm/addí
=feed_forward_sub_net_6/batch_normalization_36/batchnorm/RsqrtRsqrt?feed_forward_sub_net_6/batch_normalization_36/batchnorm/add:z:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_6/batch_normalization_36/batchnorm/Rsqrt¶
;feed_forward_sub_net_6/batch_normalization_36/batchnorm/mulMulAfeed_forward_sub_net_6/batch_normalization_36/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_6/batch_normalization_36/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_6/batch_normalization_36/batchnorm/mul
=feed_forward_sub_net_6/batch_normalization_36/batchnorm/mul_1Mulinput_1?feed_forward_sub_net_6/batch_normalization_36/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2?
=feed_forward_sub_net_6/batch_normalization_36/batchnorm/mul_1¶
=feed_forward_sub_net_6/batch_normalization_36/batchnorm/mul_2MulIfeed_forward_sub_net_6/batch_normalization_36/Cast/ReadVariableOp:value:0?feed_forward_sub_net_6/batch_normalization_36/batchnorm/mul:z:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_6/batch_normalization_36/batchnorm/mul_2¶
;feed_forward_sub_net_6/batch_normalization_36/batchnorm/subSubKfeed_forward_sub_net_6/batch_normalization_36/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_6/batch_normalization_36/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_6/batch_normalization_36/batchnorm/sub½
=feed_forward_sub_net_6/batch_normalization_36/batchnorm/add_1AddV2Afeed_forward_sub_net_6/batch_normalization_36/batchnorm/mul_1:z:0?feed_forward_sub_net_6/batch_normalization_36/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2?
=feed_forward_sub_net_6/batch_normalization_36/batchnorm/add_1í
5feed_forward_sub_net_6/dense_30/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_6_dense_30_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5feed_forward_sub_net_6/dense_30/MatMul/ReadVariableOp
&feed_forward_sub_net_6/dense_30/MatMulMatMulAfeed_forward_sub_net_6/batch_normalization_36/batchnorm/add_1:z:0=feed_forward_sub_net_6/dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&feed_forward_sub_net_6/dense_30/MatMul
Afeed_forward_sub_net_6/batch_normalization_37/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_6_batch_normalization_37_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_6/batch_normalization_37/Cast/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_37/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_37_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_37/Cast_1/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_37/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_37_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_37/Cast_2/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_37/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_37_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_37/Cast_3/ReadVariableOpÇ
=feed_forward_sub_net_6/batch_normalization_37/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2?
=feed_forward_sub_net_6/batch_normalization_37/batchnorm/add/y½
;feed_forward_sub_net_6/batch_normalization_37/batchnorm/addAddV2Kfeed_forward_sub_net_6/batch_normalization_37/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_6/batch_normalization_37/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_6/batch_normalization_37/batchnorm/addí
=feed_forward_sub_net_6/batch_normalization_37/batchnorm/RsqrtRsqrt?feed_forward_sub_net_6/batch_normalization_37/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_6/batch_normalization_37/batchnorm/Rsqrt¶
;feed_forward_sub_net_6/batch_normalization_37/batchnorm/mulMulAfeed_forward_sub_net_6/batch_normalization_37/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_6/batch_normalization_37/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_6/batch_normalization_37/batchnorm/mulª
=feed_forward_sub_net_6/batch_normalization_37/batchnorm/mul_1Mul0feed_forward_sub_net_6/dense_30/MatMul:product:0?feed_forward_sub_net_6/batch_normalization_37/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_6/batch_normalization_37/batchnorm/mul_1¶
=feed_forward_sub_net_6/batch_normalization_37/batchnorm/mul_2MulIfeed_forward_sub_net_6/batch_normalization_37/Cast/ReadVariableOp:value:0?feed_forward_sub_net_6/batch_normalization_37/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_6/batch_normalization_37/batchnorm/mul_2¶
;feed_forward_sub_net_6/batch_normalization_37/batchnorm/subSubKfeed_forward_sub_net_6/batch_normalization_37/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_6/batch_normalization_37/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_6/batch_normalization_37/batchnorm/sub½
=feed_forward_sub_net_6/batch_normalization_37/batchnorm/add_1AddV2Afeed_forward_sub_net_6/batch_normalization_37/batchnorm/mul_1:z:0?feed_forward_sub_net_6/batch_normalization_37/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_6/batch_normalization_37/batchnorm/add_1·
feed_forward_sub_net_6/ReluReluAfeed_forward_sub_net_6/batch_normalization_37/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
feed_forward_sub_net_6/Reluí
5feed_forward_sub_net_6/dense_31/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_6_dense_31_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_6/dense_31/MatMul/ReadVariableOpö
&feed_forward_sub_net_6/dense_31/MatMulMatMul)feed_forward_sub_net_6/Relu:activations:0=feed_forward_sub_net_6/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&feed_forward_sub_net_6/dense_31/MatMul
Afeed_forward_sub_net_6/batch_normalization_38/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_6_batch_normalization_38_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_6/batch_normalization_38/Cast/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_38/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_38_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_38/Cast_1/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_38/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_38_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_38/Cast_2/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_38/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_38_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_38/Cast_3/ReadVariableOpÇ
=feed_forward_sub_net_6/batch_normalization_38/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2?
=feed_forward_sub_net_6/batch_normalization_38/batchnorm/add/y½
;feed_forward_sub_net_6/batch_normalization_38/batchnorm/addAddV2Kfeed_forward_sub_net_6/batch_normalization_38/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_6/batch_normalization_38/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_6/batch_normalization_38/batchnorm/addí
=feed_forward_sub_net_6/batch_normalization_38/batchnorm/RsqrtRsqrt?feed_forward_sub_net_6/batch_normalization_38/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_6/batch_normalization_38/batchnorm/Rsqrt¶
;feed_forward_sub_net_6/batch_normalization_38/batchnorm/mulMulAfeed_forward_sub_net_6/batch_normalization_38/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_6/batch_normalization_38/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_6/batch_normalization_38/batchnorm/mulª
=feed_forward_sub_net_6/batch_normalization_38/batchnorm/mul_1Mul0feed_forward_sub_net_6/dense_31/MatMul:product:0?feed_forward_sub_net_6/batch_normalization_38/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_6/batch_normalization_38/batchnorm/mul_1¶
=feed_forward_sub_net_6/batch_normalization_38/batchnorm/mul_2MulIfeed_forward_sub_net_6/batch_normalization_38/Cast/ReadVariableOp:value:0?feed_forward_sub_net_6/batch_normalization_38/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_6/batch_normalization_38/batchnorm/mul_2¶
;feed_forward_sub_net_6/batch_normalization_38/batchnorm/subSubKfeed_forward_sub_net_6/batch_normalization_38/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_6/batch_normalization_38/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_6/batch_normalization_38/batchnorm/sub½
=feed_forward_sub_net_6/batch_normalization_38/batchnorm/add_1AddV2Afeed_forward_sub_net_6/batch_normalization_38/batchnorm/mul_1:z:0?feed_forward_sub_net_6/batch_normalization_38/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_6/batch_normalization_38/batchnorm/add_1»
feed_forward_sub_net_6/Relu_1ReluAfeed_forward_sub_net_6/batch_normalization_38/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
feed_forward_sub_net_6/Relu_1í
5feed_forward_sub_net_6/dense_32/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_6_dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_6/dense_32/MatMul/ReadVariableOpø
&feed_forward_sub_net_6/dense_32/MatMulMatMul+feed_forward_sub_net_6/Relu_1:activations:0=feed_forward_sub_net_6/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&feed_forward_sub_net_6/dense_32/MatMul
Afeed_forward_sub_net_6/batch_normalization_39/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_6_batch_normalization_39_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_6/batch_normalization_39/Cast/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_39/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_39_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_39/Cast_1/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_39/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_39_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_39/Cast_2/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_39/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_39_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_39/Cast_3/ReadVariableOpÇ
=feed_forward_sub_net_6/batch_normalization_39/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2?
=feed_forward_sub_net_6/batch_normalization_39/batchnorm/add/y½
;feed_forward_sub_net_6/batch_normalization_39/batchnorm/addAddV2Kfeed_forward_sub_net_6/batch_normalization_39/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_6/batch_normalization_39/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_6/batch_normalization_39/batchnorm/addí
=feed_forward_sub_net_6/batch_normalization_39/batchnorm/RsqrtRsqrt?feed_forward_sub_net_6/batch_normalization_39/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_6/batch_normalization_39/batchnorm/Rsqrt¶
;feed_forward_sub_net_6/batch_normalization_39/batchnorm/mulMulAfeed_forward_sub_net_6/batch_normalization_39/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_6/batch_normalization_39/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_6/batch_normalization_39/batchnorm/mulª
=feed_forward_sub_net_6/batch_normalization_39/batchnorm/mul_1Mul0feed_forward_sub_net_6/dense_32/MatMul:product:0?feed_forward_sub_net_6/batch_normalization_39/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_6/batch_normalization_39/batchnorm/mul_1¶
=feed_forward_sub_net_6/batch_normalization_39/batchnorm/mul_2MulIfeed_forward_sub_net_6/batch_normalization_39/Cast/ReadVariableOp:value:0?feed_forward_sub_net_6/batch_normalization_39/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_6/batch_normalization_39/batchnorm/mul_2¶
;feed_forward_sub_net_6/batch_normalization_39/batchnorm/subSubKfeed_forward_sub_net_6/batch_normalization_39/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_6/batch_normalization_39/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_6/batch_normalization_39/batchnorm/sub½
=feed_forward_sub_net_6/batch_normalization_39/batchnorm/add_1AddV2Afeed_forward_sub_net_6/batch_normalization_39/batchnorm/mul_1:z:0?feed_forward_sub_net_6/batch_normalization_39/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_6/batch_normalization_39/batchnorm/add_1»
feed_forward_sub_net_6/Relu_2ReluAfeed_forward_sub_net_6/batch_normalization_39/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
feed_forward_sub_net_6/Relu_2í
5feed_forward_sub_net_6/dense_33/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_6_dense_33_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_6/dense_33/MatMul/ReadVariableOpø
&feed_forward_sub_net_6/dense_33/MatMulMatMul+feed_forward_sub_net_6/Relu_2:activations:0=feed_forward_sub_net_6/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&feed_forward_sub_net_6/dense_33/MatMul
Afeed_forward_sub_net_6/batch_normalization_40/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_6_batch_normalization_40_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_6/batch_normalization_40/Cast/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_40/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_40_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_40/Cast_1/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_40/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_40_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_40/Cast_2/ReadVariableOp
Cfeed_forward_sub_net_6/batch_normalization_40/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_6_batch_normalization_40_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_6/batch_normalization_40/Cast_3/ReadVariableOpÇ
=feed_forward_sub_net_6/batch_normalization_40/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2?
=feed_forward_sub_net_6/batch_normalization_40/batchnorm/add/y½
;feed_forward_sub_net_6/batch_normalization_40/batchnorm/addAddV2Kfeed_forward_sub_net_6/batch_normalization_40/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_6/batch_normalization_40/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_6/batch_normalization_40/batchnorm/addí
=feed_forward_sub_net_6/batch_normalization_40/batchnorm/RsqrtRsqrt?feed_forward_sub_net_6/batch_normalization_40/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_6/batch_normalization_40/batchnorm/Rsqrt¶
;feed_forward_sub_net_6/batch_normalization_40/batchnorm/mulMulAfeed_forward_sub_net_6/batch_normalization_40/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_6/batch_normalization_40/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_6/batch_normalization_40/batchnorm/mulª
=feed_forward_sub_net_6/batch_normalization_40/batchnorm/mul_1Mul0feed_forward_sub_net_6/dense_33/MatMul:product:0?feed_forward_sub_net_6/batch_normalization_40/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_6/batch_normalization_40/batchnorm/mul_1¶
=feed_forward_sub_net_6/batch_normalization_40/batchnorm/mul_2MulIfeed_forward_sub_net_6/batch_normalization_40/Cast/ReadVariableOp:value:0?feed_forward_sub_net_6/batch_normalization_40/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_6/batch_normalization_40/batchnorm/mul_2¶
;feed_forward_sub_net_6/batch_normalization_40/batchnorm/subSubKfeed_forward_sub_net_6/batch_normalization_40/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_6/batch_normalization_40/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_6/batch_normalization_40/batchnorm/sub½
=feed_forward_sub_net_6/batch_normalization_40/batchnorm/add_1AddV2Afeed_forward_sub_net_6/batch_normalization_40/batchnorm/mul_1:z:0?feed_forward_sub_net_6/batch_normalization_40/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_6/batch_normalization_40/batchnorm/add_1»
feed_forward_sub_net_6/Relu_3ReluAfeed_forward_sub_net_6/batch_normalization_40/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
feed_forward_sub_net_6/Relu_3í
5feed_forward_sub_net_6/dense_34/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_6_dense_34_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5feed_forward_sub_net_6/dense_34/MatMul/ReadVariableOpø
&feed_forward_sub_net_6/dense_34/MatMulMatMul+feed_forward_sub_net_6/Relu_3:activations:0=feed_forward_sub_net_6/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&feed_forward_sub_net_6/dense_34/MatMulì
6feed_forward_sub_net_6/dense_34/BiasAdd/ReadVariableOpReadVariableOp?feed_forward_sub_net_6_dense_34_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype028
6feed_forward_sub_net_6/dense_34/BiasAdd/ReadVariableOp
'feed_forward_sub_net_6/dense_34/BiasAddBiasAdd0feed_forward_sub_net_6/dense_34/MatMul:product:0>feed_forward_sub_net_6/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2)
'feed_forward_sub_net_6/dense_34/BiasAdd
IdentityIdentity0feed_forward_sub_net_6/dense_34/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity
NoOpNoOpB^feed_forward_sub_net_6/batch_normalization_36/Cast/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_36/Cast_1/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_36/Cast_2/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_36/Cast_3/ReadVariableOpB^feed_forward_sub_net_6/batch_normalization_37/Cast/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_37/Cast_1/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_37/Cast_2/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_37/Cast_3/ReadVariableOpB^feed_forward_sub_net_6/batch_normalization_38/Cast/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_38/Cast_1/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_38/Cast_2/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_38/Cast_3/ReadVariableOpB^feed_forward_sub_net_6/batch_normalization_39/Cast/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_39/Cast_1/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_39/Cast_2/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_39/Cast_3/ReadVariableOpB^feed_forward_sub_net_6/batch_normalization_40/Cast/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_40/Cast_1/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_40/Cast_2/ReadVariableOpD^feed_forward_sub_net_6/batch_normalization_40/Cast_3/ReadVariableOp6^feed_forward_sub_net_6/dense_30/MatMul/ReadVariableOp6^feed_forward_sub_net_6/dense_31/MatMul/ReadVariableOp6^feed_forward_sub_net_6/dense_32/MatMul/ReadVariableOp6^feed_forward_sub_net_6/dense_33/MatMul/ReadVariableOp7^feed_forward_sub_net_6/dense_34/BiasAdd/ReadVariableOp6^feed_forward_sub_net_6/dense_34/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2
Afeed_forward_sub_net_6/batch_normalization_36/Cast/ReadVariableOpAfeed_forward_sub_net_6/batch_normalization_36/Cast/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_36/Cast_1/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_36/Cast_1/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_36/Cast_2/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_36/Cast_2/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_36/Cast_3/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_36/Cast_3/ReadVariableOp2
Afeed_forward_sub_net_6/batch_normalization_37/Cast/ReadVariableOpAfeed_forward_sub_net_6/batch_normalization_37/Cast/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_37/Cast_1/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_37/Cast_1/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_37/Cast_2/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_37/Cast_2/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_37/Cast_3/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_37/Cast_3/ReadVariableOp2
Afeed_forward_sub_net_6/batch_normalization_38/Cast/ReadVariableOpAfeed_forward_sub_net_6/batch_normalization_38/Cast/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_38/Cast_1/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_38/Cast_1/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_38/Cast_2/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_38/Cast_2/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_38/Cast_3/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_38/Cast_3/ReadVariableOp2
Afeed_forward_sub_net_6/batch_normalization_39/Cast/ReadVariableOpAfeed_forward_sub_net_6/batch_normalization_39/Cast/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_39/Cast_1/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_39/Cast_1/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_39/Cast_2/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_39/Cast_2/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_39/Cast_3/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_39/Cast_3/ReadVariableOp2
Afeed_forward_sub_net_6/batch_normalization_40/Cast/ReadVariableOpAfeed_forward_sub_net_6/batch_normalization_40/Cast/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_40/Cast_1/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_40/Cast_1/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_40/Cast_2/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_40/Cast_2/ReadVariableOp2
Cfeed_forward_sub_net_6/batch_normalization_40/Cast_3/ReadVariableOpCfeed_forward_sub_net_6/batch_normalization_40/Cast_3/ReadVariableOp2n
5feed_forward_sub_net_6/dense_30/MatMul/ReadVariableOp5feed_forward_sub_net_6/dense_30/MatMul/ReadVariableOp2n
5feed_forward_sub_net_6/dense_31/MatMul/ReadVariableOp5feed_forward_sub_net_6/dense_31/MatMul/ReadVariableOp2n
5feed_forward_sub_net_6/dense_32/MatMul/ReadVariableOp5feed_forward_sub_net_6/dense_32/MatMul/ReadVariableOp2n
5feed_forward_sub_net_6/dense_33/MatMul/ReadVariableOp5feed_forward_sub_net_6/dense_33/MatMul/ReadVariableOp2p
6feed_forward_sub_net_6/dense_34/BiasAdd/ReadVariableOp6feed_forward_sub_net_6/dense_34/BiasAdd/ReadVariableOp2n
5feed_forward_sub_net_6/dense_34/MatMul/ReadVariableOp5feed_forward_sub_net_6/dense_34/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
×
Ò
7__inference_batch_normalization_38_layer_call_fn_323993

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_3218692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â

7__inference_feed_forward_sub_net_6_layer_call_fn_323690
x
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:


unknown_24:

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
:ÿÿÿÿÿÿÿÿÿ
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_3226242
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
þ
­
D__inference_dense_33_layer_call_and_return_conditional_losses_322367

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
­
D__inference_dense_31_layer_call_and_return_conditional_losses_324178

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_321807

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_321973

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
­
D__inference_dense_32_layer_call_and_return_conditional_losses_322346

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»©

R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_323333
input_1A
3batch_normalization_36_cast_readvariableop_resource:
C
5batch_normalization_36_cast_1_readvariableop_resource:
C
5batch_normalization_36_cast_2_readvariableop_resource:
C
5batch_normalization_36_cast_3_readvariableop_resource:
9
'dense_30_matmul_readvariableop_resource:
A
3batch_normalization_37_cast_readvariableop_resource:C
5batch_normalization_37_cast_1_readvariableop_resource:C
5batch_normalization_37_cast_2_readvariableop_resource:C
5batch_normalization_37_cast_3_readvariableop_resource:9
'dense_31_matmul_readvariableop_resource:A
3batch_normalization_38_cast_readvariableop_resource:C
5batch_normalization_38_cast_1_readvariableop_resource:C
5batch_normalization_38_cast_2_readvariableop_resource:C
5batch_normalization_38_cast_3_readvariableop_resource:9
'dense_32_matmul_readvariableop_resource:A
3batch_normalization_39_cast_readvariableop_resource:C
5batch_normalization_39_cast_1_readvariableop_resource:C
5batch_normalization_39_cast_2_readvariableop_resource:C
5batch_normalization_39_cast_3_readvariableop_resource:9
'dense_33_matmul_readvariableop_resource:A
3batch_normalization_40_cast_readvariableop_resource:C
5batch_normalization_40_cast_1_readvariableop_resource:C
5batch_normalization_40_cast_2_readvariableop_resource:C
5batch_normalization_40_cast_3_readvariableop_resource:9
'dense_34_matmul_readvariableop_resource:
6
(dense_34_biasadd_readvariableop_resource:

identity¢*batch_normalization_36/Cast/ReadVariableOp¢,batch_normalization_36/Cast_1/ReadVariableOp¢,batch_normalization_36/Cast_2/ReadVariableOp¢,batch_normalization_36/Cast_3/ReadVariableOp¢*batch_normalization_37/Cast/ReadVariableOp¢,batch_normalization_37/Cast_1/ReadVariableOp¢,batch_normalization_37/Cast_2/ReadVariableOp¢,batch_normalization_37/Cast_3/ReadVariableOp¢*batch_normalization_38/Cast/ReadVariableOp¢,batch_normalization_38/Cast_1/ReadVariableOp¢,batch_normalization_38/Cast_2/ReadVariableOp¢,batch_normalization_38/Cast_3/ReadVariableOp¢*batch_normalization_39/Cast/ReadVariableOp¢,batch_normalization_39/Cast_1/ReadVariableOp¢,batch_normalization_39/Cast_2/ReadVariableOp¢,batch_normalization_39/Cast_3/ReadVariableOp¢*batch_normalization_40/Cast/ReadVariableOp¢,batch_normalization_40/Cast_1/ReadVariableOp¢,batch_normalization_40/Cast_2/ReadVariableOp¢,batch_normalization_40/Cast_3/ReadVariableOp¢dense_30/MatMul/ReadVariableOp¢dense_31/MatMul/ReadVariableOp¢dense_32/MatMul/ReadVariableOp¢dense_33/MatMul/ReadVariableOp¢dense_34/BiasAdd/ReadVariableOp¢dense_34/MatMul/ReadVariableOpÈ
*batch_normalization_36/Cast/ReadVariableOpReadVariableOp3batch_normalization_36_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_36/Cast/ReadVariableOpÎ
,batch_normalization_36/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_36_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_36/Cast_1/ReadVariableOpÎ
,batch_normalization_36/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_36_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_36/Cast_2/ReadVariableOpÎ
,batch_normalization_36/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_36_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_36/Cast_3/ReadVariableOp
&batch_normalization_36/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_36/batchnorm/add/yá
$batch_normalization_36/batchnorm/addAddV24batch_normalization_36/Cast_1/ReadVariableOp:value:0/batch_normalization_36/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_36/batchnorm/add¨
&batch_normalization_36/batchnorm/RsqrtRsqrt(batch_normalization_36/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_36/batchnorm/RsqrtÚ
$batch_normalization_36/batchnorm/mulMul*batch_normalization_36/batchnorm/Rsqrt:y:04batch_normalization_36/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_36/batchnorm/mul¼
&batch_normalization_36/batchnorm/mul_1Mulinput_1(batch_normalization_36/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_36/batchnorm/mul_1Ú
&batch_normalization_36/batchnorm/mul_2Mul2batch_normalization_36/Cast/ReadVariableOp:value:0(batch_normalization_36/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_36/batchnorm/mul_2Ú
$batch_normalization_36/batchnorm/subSub4batch_normalization_36/Cast_2/ReadVariableOp:value:0*batch_normalization_36/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_36/batchnorm/subá
&batch_normalization_36/batchnorm/add_1AddV2*batch_normalization_36/batchnorm/mul_1:z:0(batch_normalization_36/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_36/batchnorm/add_1¨
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_30/MatMul/ReadVariableOp²
dense_30/MatMulMatMul*batch_normalization_36/batchnorm/add_1:z:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_30/MatMulÈ
*batch_normalization_37/Cast/ReadVariableOpReadVariableOp3batch_normalization_37_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_37/Cast/ReadVariableOpÎ
,batch_normalization_37/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_37_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_37/Cast_1/ReadVariableOpÎ
,batch_normalization_37/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_37_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_37/Cast_2/ReadVariableOpÎ
,batch_normalization_37/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_37_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_37/Cast_3/ReadVariableOp
&batch_normalization_37/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_37/batchnorm/add/yá
$batch_normalization_37/batchnorm/addAddV24batch_normalization_37/Cast_1/ReadVariableOp:value:0/batch_normalization_37/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/add¨
&batch_normalization_37/batchnorm/RsqrtRsqrt(batch_normalization_37/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_37/batchnorm/RsqrtÚ
$batch_normalization_37/batchnorm/mulMul*batch_normalization_37/batchnorm/Rsqrt:y:04batch_normalization_37/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/mulÎ
&batch_normalization_37/batchnorm/mul_1Muldense_30/MatMul:product:0(batch_normalization_37/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_37/batchnorm/mul_1Ú
&batch_normalization_37/batchnorm/mul_2Mul2batch_normalization_37/Cast/ReadVariableOp:value:0(batch_normalization_37/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_37/batchnorm/mul_2Ú
$batch_normalization_37/batchnorm/subSub4batch_normalization_37/Cast_2/ReadVariableOp:value:0*batch_normalization_37/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/subá
&batch_normalization_37/batchnorm/add_1AddV2*batch_normalization_37/batchnorm/mul_1:z:0(batch_normalization_37/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_37/batchnorm/add_1r
ReluRelu*batch_normalization_37/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¨
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_31/MatMul/ReadVariableOp
dense_31/MatMulMatMulRelu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/MatMulÈ
*batch_normalization_38/Cast/ReadVariableOpReadVariableOp3batch_normalization_38_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_38/Cast/ReadVariableOpÎ
,batch_normalization_38/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_38_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_38/Cast_1/ReadVariableOpÎ
,batch_normalization_38/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_38_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_38/Cast_2/ReadVariableOpÎ
,batch_normalization_38/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_38_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_38/Cast_3/ReadVariableOp
&batch_normalization_38/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_38/batchnorm/add/yá
$batch_normalization_38/batchnorm/addAddV24batch_normalization_38/Cast_1/ReadVariableOp:value:0/batch_normalization_38/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/add¨
&batch_normalization_38/batchnorm/RsqrtRsqrt(batch_normalization_38/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_38/batchnorm/RsqrtÚ
$batch_normalization_38/batchnorm/mulMul*batch_normalization_38/batchnorm/Rsqrt:y:04batch_normalization_38/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/mulÎ
&batch_normalization_38/batchnorm/mul_1Muldense_31/MatMul:product:0(batch_normalization_38/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_38/batchnorm/mul_1Ú
&batch_normalization_38/batchnorm/mul_2Mul2batch_normalization_38/Cast/ReadVariableOp:value:0(batch_normalization_38/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_38/batchnorm/mul_2Ú
$batch_normalization_38/batchnorm/subSub4batch_normalization_38/Cast_2/ReadVariableOp:value:0*batch_normalization_38/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/subá
&batch_normalization_38/batchnorm/add_1AddV2*batch_normalization_38/batchnorm/mul_1:z:0(batch_normalization_38/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_38/batchnorm/add_1v
Relu_1Relu*batch_normalization_38/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1¨
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_32/MatMul/ReadVariableOp
dense_32/MatMulMatMulRelu_1:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_32/MatMulÈ
*batch_normalization_39/Cast/ReadVariableOpReadVariableOp3batch_normalization_39_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_39/Cast/ReadVariableOpÎ
,batch_normalization_39/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_39_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_39/Cast_1/ReadVariableOpÎ
,batch_normalization_39/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_39_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_39/Cast_2/ReadVariableOpÎ
,batch_normalization_39/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_39_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_39/Cast_3/ReadVariableOp
&batch_normalization_39/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_39/batchnorm/add/yá
$batch_normalization_39/batchnorm/addAddV24batch_normalization_39/Cast_1/ReadVariableOp:value:0/batch_normalization_39/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/add¨
&batch_normalization_39/batchnorm/RsqrtRsqrt(batch_normalization_39/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_39/batchnorm/RsqrtÚ
$batch_normalization_39/batchnorm/mulMul*batch_normalization_39/batchnorm/Rsqrt:y:04batch_normalization_39/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/mulÎ
&batch_normalization_39/batchnorm/mul_1Muldense_32/MatMul:product:0(batch_normalization_39/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_39/batchnorm/mul_1Ú
&batch_normalization_39/batchnorm/mul_2Mul2batch_normalization_39/Cast/ReadVariableOp:value:0(batch_normalization_39/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_39/batchnorm/mul_2Ú
$batch_normalization_39/batchnorm/subSub4batch_normalization_39/Cast_2/ReadVariableOp:value:0*batch_normalization_39/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/subá
&batch_normalization_39/batchnorm/add_1AddV2*batch_normalization_39/batchnorm/mul_1:z:0(batch_normalization_39/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_39/batchnorm/add_1v
Relu_2Relu*batch_normalization_39/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¨
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_33/MatMul/ReadVariableOp
dense_33/MatMulMatMulRelu_2:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_33/MatMulÈ
*batch_normalization_40/Cast/ReadVariableOpReadVariableOp3batch_normalization_40_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_40/Cast/ReadVariableOpÎ
,batch_normalization_40/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_40_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_40/Cast_1/ReadVariableOpÎ
,batch_normalization_40/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_40_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_40/Cast_2/ReadVariableOpÎ
,batch_normalization_40/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_40_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_40/Cast_3/ReadVariableOp
&batch_normalization_40/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_40/batchnorm/add/yá
$batch_normalization_40/batchnorm/addAddV24batch_normalization_40/Cast_1/ReadVariableOp:value:0/batch_normalization_40/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/add¨
&batch_normalization_40/batchnorm/RsqrtRsqrt(batch_normalization_40/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_40/batchnorm/RsqrtÚ
$batch_normalization_40/batchnorm/mulMul*batch_normalization_40/batchnorm/Rsqrt:y:04batch_normalization_40/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/mulÎ
&batch_normalization_40/batchnorm/mul_1Muldense_33/MatMul:product:0(batch_normalization_40/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_40/batchnorm/mul_1Ú
&batch_normalization_40/batchnorm/mul_2Mul2batch_normalization_40/Cast/ReadVariableOp:value:0(batch_normalization_40/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_40/batchnorm/mul_2Ú
$batch_normalization_40/batchnorm/subSub4batch_normalization_40/Cast_2/ReadVariableOp:value:0*batch_normalization_40/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/subá
&batch_normalization_40/batchnorm/add_1AddV2*batch_normalization_40/batchnorm/mul_1:z:0(batch_normalization_40/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_40/batchnorm/add_1v
Relu_3Relu*batch_normalization_40/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¨
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_34/MatMul/ReadVariableOp
dense_34/MatMulMatMulRelu_3:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_34/MatMul§
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_34/BiasAdd/ReadVariableOp¥
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_34/BiasAddt
IdentityIdentitydense_34/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity·	
NoOpNoOp+^batch_normalization_36/Cast/ReadVariableOp-^batch_normalization_36/Cast_1/ReadVariableOp-^batch_normalization_36/Cast_2/ReadVariableOp-^batch_normalization_36/Cast_3/ReadVariableOp+^batch_normalization_37/Cast/ReadVariableOp-^batch_normalization_37/Cast_1/ReadVariableOp-^batch_normalization_37/Cast_2/ReadVariableOp-^batch_normalization_37/Cast_3/ReadVariableOp+^batch_normalization_38/Cast/ReadVariableOp-^batch_normalization_38/Cast_1/ReadVariableOp-^batch_normalization_38/Cast_2/ReadVariableOp-^batch_normalization_38/Cast_3/ReadVariableOp+^batch_normalization_39/Cast/ReadVariableOp-^batch_normalization_39/Cast_1/ReadVariableOp-^batch_normalization_39/Cast_2/ReadVariableOp-^batch_normalization_39/Cast_3/ReadVariableOp+^batch_normalization_40/Cast/ReadVariableOp-^batch_normalization_40/Cast_1/ReadVariableOp-^batch_normalization_40/Cast_2/ReadVariableOp-^batch_normalization_40/Cast_3/ReadVariableOp^dense_30/MatMul/ReadVariableOp^dense_31/MatMul/ReadVariableOp^dense_32/MatMul/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_36/Cast/ReadVariableOp*batch_normalization_36/Cast/ReadVariableOp2\
,batch_normalization_36/Cast_1/ReadVariableOp,batch_normalization_36/Cast_1/ReadVariableOp2\
,batch_normalization_36/Cast_2/ReadVariableOp,batch_normalization_36/Cast_2/ReadVariableOp2\
,batch_normalization_36/Cast_3/ReadVariableOp,batch_normalization_36/Cast_3/ReadVariableOp2X
*batch_normalization_37/Cast/ReadVariableOp*batch_normalization_37/Cast/ReadVariableOp2\
,batch_normalization_37/Cast_1/ReadVariableOp,batch_normalization_37/Cast_1/ReadVariableOp2\
,batch_normalization_37/Cast_2/ReadVariableOp,batch_normalization_37/Cast_2/ReadVariableOp2\
,batch_normalization_37/Cast_3/ReadVariableOp,batch_normalization_37/Cast_3/ReadVariableOp2X
*batch_normalization_38/Cast/ReadVariableOp*batch_normalization_38/Cast/ReadVariableOp2\
,batch_normalization_38/Cast_1/ReadVariableOp,batch_normalization_38/Cast_1/ReadVariableOp2\
,batch_normalization_38/Cast_2/ReadVariableOp,batch_normalization_38/Cast_2/ReadVariableOp2\
,batch_normalization_38/Cast_3/ReadVariableOp,batch_normalization_38/Cast_3/ReadVariableOp2X
*batch_normalization_39/Cast/ReadVariableOp*batch_normalization_39/Cast/ReadVariableOp2\
,batch_normalization_39/Cast_1/ReadVariableOp,batch_normalization_39/Cast_1/ReadVariableOp2\
,batch_normalization_39/Cast_2/ReadVariableOp,batch_normalization_39/Cast_2/ReadVariableOp2\
,batch_normalization_39/Cast_3/ReadVariableOp,batch_normalization_39/Cast_3/ReadVariableOp2X
*batch_normalization_40/Cast/ReadVariableOp*batch_normalization_40/Cast/ReadVariableOp2\
,batch_normalization_40/Cast_1/ReadVariableOp,batch_normalization_40/Cast_1/ReadVariableOp2\
,batch_normalization_40/Cast_2/ReadVariableOp,batch_normalization_40/Cast_2/ReadVariableOp2\
,batch_normalization_40/Cast_3/ReadVariableOp,batch_normalization_40/Cast_3/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
È
}
)__inference_dense_31_layer_call_fn_324185

inputs
unknown:
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_3223252
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ò
7__inference_batch_normalization_39_layer_call_fn_324062

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_3219732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü

R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_323519
input_1L
>batch_normalization_36_assignmovingavg_readvariableop_resource:
N
@batch_normalization_36_assignmovingavg_1_readvariableop_resource:
A
3batch_normalization_36_cast_readvariableop_resource:
C
5batch_normalization_36_cast_1_readvariableop_resource:
9
'dense_30_matmul_readvariableop_resource:
L
>batch_normalization_37_assignmovingavg_readvariableop_resource:N
@batch_normalization_37_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_37_cast_readvariableop_resource:C
5batch_normalization_37_cast_1_readvariableop_resource:9
'dense_31_matmul_readvariableop_resource:L
>batch_normalization_38_assignmovingavg_readvariableop_resource:N
@batch_normalization_38_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_38_cast_readvariableop_resource:C
5batch_normalization_38_cast_1_readvariableop_resource:9
'dense_32_matmul_readvariableop_resource:L
>batch_normalization_39_assignmovingavg_readvariableop_resource:N
@batch_normalization_39_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_39_cast_readvariableop_resource:C
5batch_normalization_39_cast_1_readvariableop_resource:9
'dense_33_matmul_readvariableop_resource:L
>batch_normalization_40_assignmovingavg_readvariableop_resource:N
@batch_normalization_40_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_40_cast_readvariableop_resource:C
5batch_normalization_40_cast_1_readvariableop_resource:9
'dense_34_matmul_readvariableop_resource:
6
(dense_34_biasadd_readvariableop_resource:

identity¢&batch_normalization_36/AssignMovingAvg¢5batch_normalization_36/AssignMovingAvg/ReadVariableOp¢(batch_normalization_36/AssignMovingAvg_1¢7batch_normalization_36/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_36/Cast/ReadVariableOp¢,batch_normalization_36/Cast_1/ReadVariableOp¢&batch_normalization_37/AssignMovingAvg¢5batch_normalization_37/AssignMovingAvg/ReadVariableOp¢(batch_normalization_37/AssignMovingAvg_1¢7batch_normalization_37/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_37/Cast/ReadVariableOp¢,batch_normalization_37/Cast_1/ReadVariableOp¢&batch_normalization_38/AssignMovingAvg¢5batch_normalization_38/AssignMovingAvg/ReadVariableOp¢(batch_normalization_38/AssignMovingAvg_1¢7batch_normalization_38/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_38/Cast/ReadVariableOp¢,batch_normalization_38/Cast_1/ReadVariableOp¢&batch_normalization_39/AssignMovingAvg¢5batch_normalization_39/AssignMovingAvg/ReadVariableOp¢(batch_normalization_39/AssignMovingAvg_1¢7batch_normalization_39/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_39/Cast/ReadVariableOp¢,batch_normalization_39/Cast_1/ReadVariableOp¢&batch_normalization_40/AssignMovingAvg¢5batch_normalization_40/AssignMovingAvg/ReadVariableOp¢(batch_normalization_40/AssignMovingAvg_1¢7batch_normalization_40/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_40/Cast/ReadVariableOp¢,batch_normalization_40/Cast_1/ReadVariableOp¢dense_30/MatMul/ReadVariableOp¢dense_31/MatMul/ReadVariableOp¢dense_32/MatMul/ReadVariableOp¢dense_33/MatMul/ReadVariableOp¢dense_34/BiasAdd/ReadVariableOp¢dense_34/MatMul/ReadVariableOp¸
5batch_normalization_36/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_36/moments/mean/reduction_indicesÕ
#batch_normalization_36/moments/meanMeaninput_1>batch_normalization_36/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_36/moments/meanÁ
+batch_normalization_36/moments/StopGradientStopGradient,batch_normalization_36/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_36/moments/StopGradientê
0batch_normalization_36/moments/SquaredDifferenceSquaredDifferenceinput_14batch_normalization_36/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
22
0batch_normalization_36/moments/SquaredDifferenceÀ
9batch_normalization_36/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_36/moments/variance/reduction_indices
'batch_normalization_36/moments/varianceMean4batch_normalization_36/moments/SquaredDifference:z:0Bbatch_normalization_36/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_36/moments/varianceÅ
&batch_normalization_36/moments/SqueezeSqueeze,batch_normalization_36/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_36/moments/SqueezeÍ
(batch_normalization_36/moments/Squeeze_1Squeeze0batch_normalization_36/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_36/moments/Squeeze_1¡
,batch_normalization_36/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_36/AssignMovingAvg/decayÉ
+batch_normalization_36/AssignMovingAvg/CastCast5batch_normalization_36/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_36/AssignMovingAvg/Casté
5batch_normalization_36/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_36_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_36/AssignMovingAvg/ReadVariableOpô
*batch_normalization_36/AssignMovingAvg/subSub=batch_normalization_36/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_36/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_36/AssignMovingAvg/subå
*batch_normalization_36/AssignMovingAvg/mulMul.batch_normalization_36/AssignMovingAvg/sub:z:0/batch_normalization_36/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_36/AssignMovingAvg/mul²
&batch_normalization_36/AssignMovingAvgAssignSubVariableOp>batch_normalization_36_assignmovingavg_readvariableop_resource.batch_normalization_36/AssignMovingAvg/mul:z:06^batch_normalization_36/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_36/AssignMovingAvg¥
.batch_normalization_36/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_36/AssignMovingAvg_1/decayÏ
-batch_normalization_36/AssignMovingAvg_1/CastCast7batch_normalization_36/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_36/AssignMovingAvg_1/Castï
7batch_normalization_36/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_36_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_36/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_36/AssignMovingAvg_1/subSub?batch_normalization_36/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_36/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_36/AssignMovingAvg_1/subí
,batch_normalization_36/AssignMovingAvg_1/mulMul0batch_normalization_36/AssignMovingAvg_1/sub:z:01batch_normalization_36/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_36/AssignMovingAvg_1/mul¼
(batch_normalization_36/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_36_assignmovingavg_1_readvariableop_resource0batch_normalization_36/AssignMovingAvg_1/mul:z:08^batch_normalization_36/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_36/AssignMovingAvg_1È
*batch_normalization_36/Cast/ReadVariableOpReadVariableOp3batch_normalization_36_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_36/Cast/ReadVariableOpÎ
,batch_normalization_36/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_36_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_36/Cast_1/ReadVariableOp
&batch_normalization_36/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_36/batchnorm/add/yÞ
$batch_normalization_36/batchnorm/addAddV21batch_normalization_36/moments/Squeeze_1:output:0/batch_normalization_36/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_36/batchnorm/add¨
&batch_normalization_36/batchnorm/RsqrtRsqrt(batch_normalization_36/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_36/batchnorm/RsqrtÚ
$batch_normalization_36/batchnorm/mulMul*batch_normalization_36/batchnorm/Rsqrt:y:04batch_normalization_36/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_36/batchnorm/mul¼
&batch_normalization_36/batchnorm/mul_1Mulinput_1(batch_normalization_36/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_36/batchnorm/mul_1×
&batch_normalization_36/batchnorm/mul_2Mul/batch_normalization_36/moments/Squeeze:output:0(batch_normalization_36/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_36/batchnorm/mul_2Ø
$batch_normalization_36/batchnorm/subSub2batch_normalization_36/Cast/ReadVariableOp:value:0*batch_normalization_36/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_36/batchnorm/subá
&batch_normalization_36/batchnorm/add_1AddV2*batch_normalization_36/batchnorm/mul_1:z:0(batch_normalization_36/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_36/batchnorm/add_1¨
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_30/MatMul/ReadVariableOp²
dense_30/MatMulMatMul*batch_normalization_36/batchnorm/add_1:z:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_30/MatMul¸
5batch_normalization_37/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_37/moments/mean/reduction_indicesç
#batch_normalization_37/moments/meanMeandense_30/MatMul:product:0>batch_normalization_37/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_37/moments/meanÁ
+batch_normalization_37/moments/StopGradientStopGradient,batch_normalization_37/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_37/moments/StopGradientü
0batch_normalization_37/moments/SquaredDifferenceSquaredDifferencedense_30/MatMul:product:04batch_normalization_37/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_37/moments/SquaredDifferenceÀ
9batch_normalization_37/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_37/moments/variance/reduction_indices
'batch_normalization_37/moments/varianceMean4batch_normalization_37/moments/SquaredDifference:z:0Bbatch_normalization_37/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_37/moments/varianceÅ
&batch_normalization_37/moments/SqueezeSqueeze,batch_normalization_37/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_37/moments/SqueezeÍ
(batch_normalization_37/moments/Squeeze_1Squeeze0batch_normalization_37/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_37/moments/Squeeze_1¡
,batch_normalization_37/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_37/AssignMovingAvg/decayÉ
+batch_normalization_37/AssignMovingAvg/CastCast5batch_normalization_37/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_37/AssignMovingAvg/Casté
5batch_normalization_37/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_37_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_37/AssignMovingAvg/ReadVariableOpô
*batch_normalization_37/AssignMovingAvg/subSub=batch_normalization_37/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_37/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_37/AssignMovingAvg/subå
*batch_normalization_37/AssignMovingAvg/mulMul.batch_normalization_37/AssignMovingAvg/sub:z:0/batch_normalization_37/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_37/AssignMovingAvg/mul²
&batch_normalization_37/AssignMovingAvgAssignSubVariableOp>batch_normalization_37_assignmovingavg_readvariableop_resource.batch_normalization_37/AssignMovingAvg/mul:z:06^batch_normalization_37/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_37/AssignMovingAvg¥
.batch_normalization_37/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_37/AssignMovingAvg_1/decayÏ
-batch_normalization_37/AssignMovingAvg_1/CastCast7batch_normalization_37/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_37/AssignMovingAvg_1/Castï
7batch_normalization_37/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_37_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_37/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_37/AssignMovingAvg_1/subSub?batch_normalization_37/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_37/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_37/AssignMovingAvg_1/subí
,batch_normalization_37/AssignMovingAvg_1/mulMul0batch_normalization_37/AssignMovingAvg_1/sub:z:01batch_normalization_37/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_37/AssignMovingAvg_1/mul¼
(batch_normalization_37/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_37_assignmovingavg_1_readvariableop_resource0batch_normalization_37/AssignMovingAvg_1/mul:z:08^batch_normalization_37/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_37/AssignMovingAvg_1È
*batch_normalization_37/Cast/ReadVariableOpReadVariableOp3batch_normalization_37_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_37/Cast/ReadVariableOpÎ
,batch_normalization_37/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_37_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_37/Cast_1/ReadVariableOp
&batch_normalization_37/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_37/batchnorm/add/yÞ
$batch_normalization_37/batchnorm/addAddV21batch_normalization_37/moments/Squeeze_1:output:0/batch_normalization_37/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/add¨
&batch_normalization_37/batchnorm/RsqrtRsqrt(batch_normalization_37/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_37/batchnorm/RsqrtÚ
$batch_normalization_37/batchnorm/mulMul*batch_normalization_37/batchnorm/Rsqrt:y:04batch_normalization_37/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/mulÎ
&batch_normalization_37/batchnorm/mul_1Muldense_30/MatMul:product:0(batch_normalization_37/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_37/batchnorm/mul_1×
&batch_normalization_37/batchnorm/mul_2Mul/batch_normalization_37/moments/Squeeze:output:0(batch_normalization_37/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_37/batchnorm/mul_2Ø
$batch_normalization_37/batchnorm/subSub2batch_normalization_37/Cast/ReadVariableOp:value:0*batch_normalization_37/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/subá
&batch_normalization_37/batchnorm/add_1AddV2*batch_normalization_37/batchnorm/mul_1:z:0(batch_normalization_37/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_37/batchnorm/add_1r
ReluRelu*batch_normalization_37/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¨
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_31/MatMul/ReadVariableOp
dense_31/MatMulMatMulRelu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/MatMul¸
5batch_normalization_38/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_38/moments/mean/reduction_indicesç
#batch_normalization_38/moments/meanMeandense_31/MatMul:product:0>batch_normalization_38/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_38/moments/meanÁ
+batch_normalization_38/moments/StopGradientStopGradient,batch_normalization_38/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_38/moments/StopGradientü
0batch_normalization_38/moments/SquaredDifferenceSquaredDifferencedense_31/MatMul:product:04batch_normalization_38/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_38/moments/SquaredDifferenceÀ
9batch_normalization_38/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_38/moments/variance/reduction_indices
'batch_normalization_38/moments/varianceMean4batch_normalization_38/moments/SquaredDifference:z:0Bbatch_normalization_38/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_38/moments/varianceÅ
&batch_normalization_38/moments/SqueezeSqueeze,batch_normalization_38/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_38/moments/SqueezeÍ
(batch_normalization_38/moments/Squeeze_1Squeeze0batch_normalization_38/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_38/moments/Squeeze_1¡
,batch_normalization_38/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_38/AssignMovingAvg/decayÉ
+batch_normalization_38/AssignMovingAvg/CastCast5batch_normalization_38/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_38/AssignMovingAvg/Casté
5batch_normalization_38/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_38_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_38/AssignMovingAvg/ReadVariableOpô
*batch_normalization_38/AssignMovingAvg/subSub=batch_normalization_38/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_38/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_38/AssignMovingAvg/subå
*batch_normalization_38/AssignMovingAvg/mulMul.batch_normalization_38/AssignMovingAvg/sub:z:0/batch_normalization_38/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_38/AssignMovingAvg/mul²
&batch_normalization_38/AssignMovingAvgAssignSubVariableOp>batch_normalization_38_assignmovingavg_readvariableop_resource.batch_normalization_38/AssignMovingAvg/mul:z:06^batch_normalization_38/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_38/AssignMovingAvg¥
.batch_normalization_38/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_38/AssignMovingAvg_1/decayÏ
-batch_normalization_38/AssignMovingAvg_1/CastCast7batch_normalization_38/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_38/AssignMovingAvg_1/Castï
7batch_normalization_38/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_38_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_38/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_38/AssignMovingAvg_1/subSub?batch_normalization_38/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_38/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_38/AssignMovingAvg_1/subí
,batch_normalization_38/AssignMovingAvg_1/mulMul0batch_normalization_38/AssignMovingAvg_1/sub:z:01batch_normalization_38/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_38/AssignMovingAvg_1/mul¼
(batch_normalization_38/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_38_assignmovingavg_1_readvariableop_resource0batch_normalization_38/AssignMovingAvg_1/mul:z:08^batch_normalization_38/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_38/AssignMovingAvg_1È
*batch_normalization_38/Cast/ReadVariableOpReadVariableOp3batch_normalization_38_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_38/Cast/ReadVariableOpÎ
,batch_normalization_38/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_38_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_38/Cast_1/ReadVariableOp
&batch_normalization_38/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_38/batchnorm/add/yÞ
$batch_normalization_38/batchnorm/addAddV21batch_normalization_38/moments/Squeeze_1:output:0/batch_normalization_38/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/add¨
&batch_normalization_38/batchnorm/RsqrtRsqrt(batch_normalization_38/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_38/batchnorm/RsqrtÚ
$batch_normalization_38/batchnorm/mulMul*batch_normalization_38/batchnorm/Rsqrt:y:04batch_normalization_38/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/mulÎ
&batch_normalization_38/batchnorm/mul_1Muldense_31/MatMul:product:0(batch_normalization_38/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_38/batchnorm/mul_1×
&batch_normalization_38/batchnorm/mul_2Mul/batch_normalization_38/moments/Squeeze:output:0(batch_normalization_38/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_38/batchnorm/mul_2Ø
$batch_normalization_38/batchnorm/subSub2batch_normalization_38/Cast/ReadVariableOp:value:0*batch_normalization_38/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/subá
&batch_normalization_38/batchnorm/add_1AddV2*batch_normalization_38/batchnorm/mul_1:z:0(batch_normalization_38/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_38/batchnorm/add_1v
Relu_1Relu*batch_normalization_38/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1¨
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_32/MatMul/ReadVariableOp
dense_32/MatMulMatMulRelu_1:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_32/MatMul¸
5batch_normalization_39/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_39/moments/mean/reduction_indicesç
#batch_normalization_39/moments/meanMeandense_32/MatMul:product:0>batch_normalization_39/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_39/moments/meanÁ
+batch_normalization_39/moments/StopGradientStopGradient,batch_normalization_39/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_39/moments/StopGradientü
0batch_normalization_39/moments/SquaredDifferenceSquaredDifferencedense_32/MatMul:product:04batch_normalization_39/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_39/moments/SquaredDifferenceÀ
9batch_normalization_39/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_39/moments/variance/reduction_indices
'batch_normalization_39/moments/varianceMean4batch_normalization_39/moments/SquaredDifference:z:0Bbatch_normalization_39/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_39/moments/varianceÅ
&batch_normalization_39/moments/SqueezeSqueeze,batch_normalization_39/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_39/moments/SqueezeÍ
(batch_normalization_39/moments/Squeeze_1Squeeze0batch_normalization_39/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_39/moments/Squeeze_1¡
,batch_normalization_39/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_39/AssignMovingAvg/decayÉ
+batch_normalization_39/AssignMovingAvg/CastCast5batch_normalization_39/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_39/AssignMovingAvg/Casté
5batch_normalization_39/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_39_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_39/AssignMovingAvg/ReadVariableOpô
*batch_normalization_39/AssignMovingAvg/subSub=batch_normalization_39/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_39/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_39/AssignMovingAvg/subå
*batch_normalization_39/AssignMovingAvg/mulMul.batch_normalization_39/AssignMovingAvg/sub:z:0/batch_normalization_39/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_39/AssignMovingAvg/mul²
&batch_normalization_39/AssignMovingAvgAssignSubVariableOp>batch_normalization_39_assignmovingavg_readvariableop_resource.batch_normalization_39/AssignMovingAvg/mul:z:06^batch_normalization_39/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_39/AssignMovingAvg¥
.batch_normalization_39/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_39/AssignMovingAvg_1/decayÏ
-batch_normalization_39/AssignMovingAvg_1/CastCast7batch_normalization_39/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_39/AssignMovingAvg_1/Castï
7batch_normalization_39/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_39_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_39/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_39/AssignMovingAvg_1/subSub?batch_normalization_39/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_39/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_39/AssignMovingAvg_1/subí
,batch_normalization_39/AssignMovingAvg_1/mulMul0batch_normalization_39/AssignMovingAvg_1/sub:z:01batch_normalization_39/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_39/AssignMovingAvg_1/mul¼
(batch_normalization_39/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_39_assignmovingavg_1_readvariableop_resource0batch_normalization_39/AssignMovingAvg_1/mul:z:08^batch_normalization_39/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_39/AssignMovingAvg_1È
*batch_normalization_39/Cast/ReadVariableOpReadVariableOp3batch_normalization_39_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_39/Cast/ReadVariableOpÎ
,batch_normalization_39/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_39_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_39/Cast_1/ReadVariableOp
&batch_normalization_39/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_39/batchnorm/add/yÞ
$batch_normalization_39/batchnorm/addAddV21batch_normalization_39/moments/Squeeze_1:output:0/batch_normalization_39/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/add¨
&batch_normalization_39/batchnorm/RsqrtRsqrt(batch_normalization_39/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_39/batchnorm/RsqrtÚ
$batch_normalization_39/batchnorm/mulMul*batch_normalization_39/batchnorm/Rsqrt:y:04batch_normalization_39/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/mulÎ
&batch_normalization_39/batchnorm/mul_1Muldense_32/MatMul:product:0(batch_normalization_39/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_39/batchnorm/mul_1×
&batch_normalization_39/batchnorm/mul_2Mul/batch_normalization_39/moments/Squeeze:output:0(batch_normalization_39/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_39/batchnorm/mul_2Ø
$batch_normalization_39/batchnorm/subSub2batch_normalization_39/Cast/ReadVariableOp:value:0*batch_normalization_39/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/subá
&batch_normalization_39/batchnorm/add_1AddV2*batch_normalization_39/batchnorm/mul_1:z:0(batch_normalization_39/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_39/batchnorm/add_1v
Relu_2Relu*batch_normalization_39/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¨
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_33/MatMul/ReadVariableOp
dense_33/MatMulMatMulRelu_2:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_33/MatMul¸
5batch_normalization_40/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_40/moments/mean/reduction_indicesç
#batch_normalization_40/moments/meanMeandense_33/MatMul:product:0>batch_normalization_40/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_40/moments/meanÁ
+batch_normalization_40/moments/StopGradientStopGradient,batch_normalization_40/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_40/moments/StopGradientü
0batch_normalization_40/moments/SquaredDifferenceSquaredDifferencedense_33/MatMul:product:04batch_normalization_40/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_40/moments/SquaredDifferenceÀ
9batch_normalization_40/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_40/moments/variance/reduction_indices
'batch_normalization_40/moments/varianceMean4batch_normalization_40/moments/SquaredDifference:z:0Bbatch_normalization_40/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_40/moments/varianceÅ
&batch_normalization_40/moments/SqueezeSqueeze,batch_normalization_40/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_40/moments/SqueezeÍ
(batch_normalization_40/moments/Squeeze_1Squeeze0batch_normalization_40/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_40/moments/Squeeze_1¡
,batch_normalization_40/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_40/AssignMovingAvg/decayÉ
+batch_normalization_40/AssignMovingAvg/CastCast5batch_normalization_40/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_40/AssignMovingAvg/Casté
5batch_normalization_40/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_40_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_40/AssignMovingAvg/ReadVariableOpô
*batch_normalization_40/AssignMovingAvg/subSub=batch_normalization_40/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_40/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_40/AssignMovingAvg/subå
*batch_normalization_40/AssignMovingAvg/mulMul.batch_normalization_40/AssignMovingAvg/sub:z:0/batch_normalization_40/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_40/AssignMovingAvg/mul²
&batch_normalization_40/AssignMovingAvgAssignSubVariableOp>batch_normalization_40_assignmovingavg_readvariableop_resource.batch_normalization_40/AssignMovingAvg/mul:z:06^batch_normalization_40/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_40/AssignMovingAvg¥
.batch_normalization_40/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_40/AssignMovingAvg_1/decayÏ
-batch_normalization_40/AssignMovingAvg_1/CastCast7batch_normalization_40/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_40/AssignMovingAvg_1/Castï
7batch_normalization_40/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_40_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_40/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_40/AssignMovingAvg_1/subSub?batch_normalization_40/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_40/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_40/AssignMovingAvg_1/subí
,batch_normalization_40/AssignMovingAvg_1/mulMul0batch_normalization_40/AssignMovingAvg_1/sub:z:01batch_normalization_40/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_40/AssignMovingAvg_1/mul¼
(batch_normalization_40/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_40_assignmovingavg_1_readvariableop_resource0batch_normalization_40/AssignMovingAvg_1/mul:z:08^batch_normalization_40/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_40/AssignMovingAvg_1È
*batch_normalization_40/Cast/ReadVariableOpReadVariableOp3batch_normalization_40_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_40/Cast/ReadVariableOpÎ
,batch_normalization_40/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_40_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_40/Cast_1/ReadVariableOp
&batch_normalization_40/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_40/batchnorm/add/yÞ
$batch_normalization_40/batchnorm/addAddV21batch_normalization_40/moments/Squeeze_1:output:0/batch_normalization_40/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/add¨
&batch_normalization_40/batchnorm/RsqrtRsqrt(batch_normalization_40/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_40/batchnorm/RsqrtÚ
$batch_normalization_40/batchnorm/mulMul*batch_normalization_40/batchnorm/Rsqrt:y:04batch_normalization_40/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/mulÎ
&batch_normalization_40/batchnorm/mul_1Muldense_33/MatMul:product:0(batch_normalization_40/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_40/batchnorm/mul_1×
&batch_normalization_40/batchnorm/mul_2Mul/batch_normalization_40/moments/Squeeze:output:0(batch_normalization_40/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_40/batchnorm/mul_2Ø
$batch_normalization_40/batchnorm/subSub2batch_normalization_40/Cast/ReadVariableOp:value:0*batch_normalization_40/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/subá
&batch_normalization_40/batchnorm/add_1AddV2*batch_normalization_40/batchnorm/mul_1:z:0(batch_normalization_40/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_40/batchnorm/add_1v
Relu_3Relu*batch_normalization_40/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¨
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_34/MatMul/ReadVariableOp
dense_34/MatMulMatMulRelu_3:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_34/MatMul§
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_34/BiasAdd/ReadVariableOp¥
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_34/BiasAddt
IdentityIdentitydense_34/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity¿
NoOpNoOp'^batch_normalization_36/AssignMovingAvg6^batch_normalization_36/AssignMovingAvg/ReadVariableOp)^batch_normalization_36/AssignMovingAvg_18^batch_normalization_36/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_36/Cast/ReadVariableOp-^batch_normalization_36/Cast_1/ReadVariableOp'^batch_normalization_37/AssignMovingAvg6^batch_normalization_37/AssignMovingAvg/ReadVariableOp)^batch_normalization_37/AssignMovingAvg_18^batch_normalization_37/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_37/Cast/ReadVariableOp-^batch_normalization_37/Cast_1/ReadVariableOp'^batch_normalization_38/AssignMovingAvg6^batch_normalization_38/AssignMovingAvg/ReadVariableOp)^batch_normalization_38/AssignMovingAvg_18^batch_normalization_38/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_38/Cast/ReadVariableOp-^batch_normalization_38/Cast_1/ReadVariableOp'^batch_normalization_39/AssignMovingAvg6^batch_normalization_39/AssignMovingAvg/ReadVariableOp)^batch_normalization_39/AssignMovingAvg_18^batch_normalization_39/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_39/Cast/ReadVariableOp-^batch_normalization_39/Cast_1/ReadVariableOp'^batch_normalization_40/AssignMovingAvg6^batch_normalization_40/AssignMovingAvg/ReadVariableOp)^batch_normalization_40/AssignMovingAvg_18^batch_normalization_40/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_40/Cast/ReadVariableOp-^batch_normalization_40/Cast_1/ReadVariableOp^dense_30/MatMul/ReadVariableOp^dense_31/MatMul/ReadVariableOp^dense_32/MatMul/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_36/AssignMovingAvg&batch_normalization_36/AssignMovingAvg2n
5batch_normalization_36/AssignMovingAvg/ReadVariableOp5batch_normalization_36/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_36/AssignMovingAvg_1(batch_normalization_36/AssignMovingAvg_12r
7batch_normalization_36/AssignMovingAvg_1/ReadVariableOp7batch_normalization_36/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_36/Cast/ReadVariableOp*batch_normalization_36/Cast/ReadVariableOp2\
,batch_normalization_36/Cast_1/ReadVariableOp,batch_normalization_36/Cast_1/ReadVariableOp2P
&batch_normalization_37/AssignMovingAvg&batch_normalization_37/AssignMovingAvg2n
5batch_normalization_37/AssignMovingAvg/ReadVariableOp5batch_normalization_37/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_37/AssignMovingAvg_1(batch_normalization_37/AssignMovingAvg_12r
7batch_normalization_37/AssignMovingAvg_1/ReadVariableOp7batch_normalization_37/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_37/Cast/ReadVariableOp*batch_normalization_37/Cast/ReadVariableOp2\
,batch_normalization_37/Cast_1/ReadVariableOp,batch_normalization_37/Cast_1/ReadVariableOp2P
&batch_normalization_38/AssignMovingAvg&batch_normalization_38/AssignMovingAvg2n
5batch_normalization_38/AssignMovingAvg/ReadVariableOp5batch_normalization_38/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_38/AssignMovingAvg_1(batch_normalization_38/AssignMovingAvg_12r
7batch_normalization_38/AssignMovingAvg_1/ReadVariableOp7batch_normalization_38/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_38/Cast/ReadVariableOp*batch_normalization_38/Cast/ReadVariableOp2\
,batch_normalization_38/Cast_1/ReadVariableOp,batch_normalization_38/Cast_1/ReadVariableOp2P
&batch_normalization_39/AssignMovingAvg&batch_normalization_39/AssignMovingAvg2n
5batch_normalization_39/AssignMovingAvg/ReadVariableOp5batch_normalization_39/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_39/AssignMovingAvg_1(batch_normalization_39/AssignMovingAvg_12r
7batch_normalization_39/AssignMovingAvg_1/ReadVariableOp7batch_normalization_39/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_39/Cast/ReadVariableOp*batch_normalization_39/Cast/ReadVariableOp2\
,batch_normalization_39/Cast_1/ReadVariableOp,batch_normalization_39/Cast_1/ReadVariableOp2P
&batch_normalization_40/AssignMovingAvg&batch_normalization_40/AssignMovingAvg2n
5batch_normalization_40/AssignMovingAvg/ReadVariableOp5batch_normalization_40/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_40/AssignMovingAvg_1(batch_normalization_40/AssignMovingAvg_12r
7batch_normalization_40/AssignMovingAvg_1/ReadVariableOp7batch_normalization_40/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_40/Cast/ReadVariableOp*batch_normalization_40/Cast/ReadVariableOp2\
,batch_normalization_40/Cast_1/ReadVariableOp,batch_normalization_40/Cast_1/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
Óz
Ò
"__inference__traced_restore_324421
file_prefixR
Dassignvariableop_feed_forward_sub_net_6_batch_normalization_36_gamma:
S
Eassignvariableop_1_feed_forward_sub_net_6_batch_normalization_36_beta:
T
Fassignvariableop_2_feed_forward_sub_net_6_batch_normalization_37_gamma:S
Eassignvariableop_3_feed_forward_sub_net_6_batch_normalization_37_beta:T
Fassignvariableop_4_feed_forward_sub_net_6_batch_normalization_38_gamma:S
Eassignvariableop_5_feed_forward_sub_net_6_batch_normalization_38_beta:T
Fassignvariableop_6_feed_forward_sub_net_6_batch_normalization_39_gamma:S
Eassignvariableop_7_feed_forward_sub_net_6_batch_normalization_39_beta:T
Fassignvariableop_8_feed_forward_sub_net_6_batch_normalization_40_gamma:S
Eassignvariableop_9_feed_forward_sub_net_6_batch_normalization_40_beta:[
Massignvariableop_10_feed_forward_sub_net_6_batch_normalization_36_moving_mean:
_
Qassignvariableop_11_feed_forward_sub_net_6_batch_normalization_36_moving_variance:
[
Massignvariableop_12_feed_forward_sub_net_6_batch_normalization_37_moving_mean:_
Qassignvariableop_13_feed_forward_sub_net_6_batch_normalization_37_moving_variance:[
Massignvariableop_14_feed_forward_sub_net_6_batch_normalization_38_moving_mean:_
Qassignvariableop_15_feed_forward_sub_net_6_batch_normalization_38_moving_variance:[
Massignvariableop_16_feed_forward_sub_net_6_batch_normalization_39_moving_mean:_
Qassignvariableop_17_feed_forward_sub_net_6_batch_normalization_39_moving_variance:[
Massignvariableop_18_feed_forward_sub_net_6_batch_normalization_40_moving_mean:_
Qassignvariableop_19_feed_forward_sub_net_6_batch_normalization_40_moving_variance:L
:assignvariableop_20_feed_forward_sub_net_6_dense_30_kernel:
L
:assignvariableop_21_feed_forward_sub_net_6_dense_31_kernel:L
:assignvariableop_22_feed_forward_sub_net_6_dense_32_kernel:L
:assignvariableop_23_feed_forward_sub_net_6_dense_33_kernel:L
:assignvariableop_24_feed_forward_sub_net_6_dense_34_kernel:
F
8assignvariableop_25_feed_forward_sub_net_6_dense_34_bias:

identity_27¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ç	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ó
valueÉBÆB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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

IdentityÃ
AssignVariableOpAssignVariableOpDassignvariableop_feed_forward_sub_net_6_batch_normalization_36_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ê
AssignVariableOp_1AssignVariableOpEassignvariableop_1_feed_forward_sub_net_6_batch_normalization_36_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ë
AssignVariableOp_2AssignVariableOpFassignvariableop_2_feed_forward_sub_net_6_batch_normalization_37_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ê
AssignVariableOp_3AssignVariableOpEassignvariableop_3_feed_forward_sub_net_6_batch_normalization_37_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ë
AssignVariableOp_4AssignVariableOpFassignvariableop_4_feed_forward_sub_net_6_batch_normalization_38_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ê
AssignVariableOp_5AssignVariableOpEassignvariableop_5_feed_forward_sub_net_6_batch_normalization_38_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ë
AssignVariableOp_6AssignVariableOpFassignvariableop_6_feed_forward_sub_net_6_batch_normalization_39_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ê
AssignVariableOp_7AssignVariableOpEassignvariableop_7_feed_forward_sub_net_6_batch_normalization_39_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ë
AssignVariableOp_8AssignVariableOpFassignvariableop_8_feed_forward_sub_net_6_batch_normalization_40_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ê
AssignVariableOp_9AssignVariableOpEassignvariableop_9_feed_forward_sub_net_6_batch_normalization_40_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Õ
AssignVariableOp_10AssignVariableOpMassignvariableop_10_feed_forward_sub_net_6_batch_normalization_36_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ù
AssignVariableOp_11AssignVariableOpQassignvariableop_11_feed_forward_sub_net_6_batch_normalization_36_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Õ
AssignVariableOp_12AssignVariableOpMassignvariableop_12_feed_forward_sub_net_6_batch_normalization_37_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ù
AssignVariableOp_13AssignVariableOpQassignvariableop_13_feed_forward_sub_net_6_batch_normalization_37_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Õ
AssignVariableOp_14AssignVariableOpMassignvariableop_14_feed_forward_sub_net_6_batch_normalization_38_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ù
AssignVariableOp_15AssignVariableOpQassignvariableop_15_feed_forward_sub_net_6_batch_normalization_38_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Õ
AssignVariableOp_16AssignVariableOpMassignvariableop_16_feed_forward_sub_net_6_batch_normalization_39_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ù
AssignVariableOp_17AssignVariableOpQassignvariableop_17_feed_forward_sub_net_6_batch_normalization_39_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Õ
AssignVariableOp_18AssignVariableOpMassignvariableop_18_feed_forward_sub_net_6_batch_normalization_40_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ù
AssignVariableOp_19AssignVariableOpQassignvariableop_19_feed_forward_sub_net_6_batch_normalization_40_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Â
AssignVariableOp_20AssignVariableOp:assignvariableop_20_feed_forward_sub_net_6_dense_30_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Â
AssignVariableOp_21AssignVariableOp:assignvariableop_21_feed_forward_sub_net_6_dense_31_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Â
AssignVariableOp_22AssignVariableOp:assignvariableop_22_feed_forward_sub_net_6_dense_32_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Â
AssignVariableOp_23AssignVariableOp:assignvariableop_23_feed_forward_sub_net_6_dense_33_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Â
AssignVariableOp_24AssignVariableOp:assignvariableop_24_feed_forward_sub_net_6_dense_34_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25À
AssignVariableOp_25AssignVariableOp8assignvariableop_25_feed_forward_sub_net_6_dense_34_biasIdentity_25:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix
þ
­
D__inference_dense_32_layer_call_and_return_conditional_losses_324192

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
}
)__inference_dense_30_layer_call_fn_324171

inputs
unknown:

identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_3223042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
þ
­
D__inference_dense_33_layer_call_and_return_conditional_losses_324206

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º

$__inference_signature_wrapper_322935
input_1
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:


unknown_24:

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
:ÿÿÿÿÿÿÿÿÿ
*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_3214512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
þ

7__inference_feed_forward_sub_net_6_layer_call_fn_323576
input_1
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:


unknown_24:

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
:ÿÿÿÿÿÿÿÿÿ
*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_3223982
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
±

R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_321641

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
­
D__inference_dense_31_layer_call_and_return_conditional_losses_322325

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
Ò
7__inference_batch_normalization_36_layer_call_fn_323829

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_3215372
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ë+
Ó
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_323885

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_323767

inputs*
cast_readvariableop_resource:
,
cast_1_readvariableop_resource:
,
cast_2_readvariableop_resource:
,
cast_3_readvariableop_resource:

identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:
*
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
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
±

R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_324095

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô

7__inference_feed_forward_sub_net_6_layer_call_fn_323747
input_1
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:


unknown_24:

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
:ÿÿÿÿÿÿÿÿÿ
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_3226242
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
×
Ò
7__inference_batch_normalization_40_layer_call_fn_324157

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_3222012
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

õ
D__inference_dense_34_layer_call_and_return_conditional_losses_322391

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
Ò
7__inference_batch_normalization_39_layer_call_fn_324075

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_3220352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ò
7__inference_batch_normalization_37_layer_call_fn_323898

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_3216412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
Ò
7__inference_batch_normalization_37_layer_call_fn_323911

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_3217032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë+
Ó
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_322201

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë+
Ó
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_321703

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_321475

inputs*
cast_readvariableop_resource:
,
cast_1_readvariableop_resource:
,
cast_2_readvariableop_resource:
,
cast_3_readvariableop_resource:

identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:
*
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
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ë+
Ó
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_321537

inputs5
'assignmovingavg_readvariableop_resource:
7
)assignmovingavg_1_readvariableop_resource:
*
cast_readvariableop_resource:
,
cast_1_readvariableop_resource:

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

:
*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
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

:
*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
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
:
*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:
2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2
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
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:
*
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
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ù
Ò
7__inference_batch_normalization_36_layer_call_fn_323816

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_3214752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
©©

R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_323041
xA
3batch_normalization_36_cast_readvariableop_resource:
C
5batch_normalization_36_cast_1_readvariableop_resource:
C
5batch_normalization_36_cast_2_readvariableop_resource:
C
5batch_normalization_36_cast_3_readvariableop_resource:
9
'dense_30_matmul_readvariableop_resource:
A
3batch_normalization_37_cast_readvariableop_resource:C
5batch_normalization_37_cast_1_readvariableop_resource:C
5batch_normalization_37_cast_2_readvariableop_resource:C
5batch_normalization_37_cast_3_readvariableop_resource:9
'dense_31_matmul_readvariableop_resource:A
3batch_normalization_38_cast_readvariableop_resource:C
5batch_normalization_38_cast_1_readvariableop_resource:C
5batch_normalization_38_cast_2_readvariableop_resource:C
5batch_normalization_38_cast_3_readvariableop_resource:9
'dense_32_matmul_readvariableop_resource:A
3batch_normalization_39_cast_readvariableop_resource:C
5batch_normalization_39_cast_1_readvariableop_resource:C
5batch_normalization_39_cast_2_readvariableop_resource:C
5batch_normalization_39_cast_3_readvariableop_resource:9
'dense_33_matmul_readvariableop_resource:A
3batch_normalization_40_cast_readvariableop_resource:C
5batch_normalization_40_cast_1_readvariableop_resource:C
5batch_normalization_40_cast_2_readvariableop_resource:C
5batch_normalization_40_cast_3_readvariableop_resource:9
'dense_34_matmul_readvariableop_resource:
6
(dense_34_biasadd_readvariableop_resource:

identity¢*batch_normalization_36/Cast/ReadVariableOp¢,batch_normalization_36/Cast_1/ReadVariableOp¢,batch_normalization_36/Cast_2/ReadVariableOp¢,batch_normalization_36/Cast_3/ReadVariableOp¢*batch_normalization_37/Cast/ReadVariableOp¢,batch_normalization_37/Cast_1/ReadVariableOp¢,batch_normalization_37/Cast_2/ReadVariableOp¢,batch_normalization_37/Cast_3/ReadVariableOp¢*batch_normalization_38/Cast/ReadVariableOp¢,batch_normalization_38/Cast_1/ReadVariableOp¢,batch_normalization_38/Cast_2/ReadVariableOp¢,batch_normalization_38/Cast_3/ReadVariableOp¢*batch_normalization_39/Cast/ReadVariableOp¢,batch_normalization_39/Cast_1/ReadVariableOp¢,batch_normalization_39/Cast_2/ReadVariableOp¢,batch_normalization_39/Cast_3/ReadVariableOp¢*batch_normalization_40/Cast/ReadVariableOp¢,batch_normalization_40/Cast_1/ReadVariableOp¢,batch_normalization_40/Cast_2/ReadVariableOp¢,batch_normalization_40/Cast_3/ReadVariableOp¢dense_30/MatMul/ReadVariableOp¢dense_31/MatMul/ReadVariableOp¢dense_32/MatMul/ReadVariableOp¢dense_33/MatMul/ReadVariableOp¢dense_34/BiasAdd/ReadVariableOp¢dense_34/MatMul/ReadVariableOpÈ
*batch_normalization_36/Cast/ReadVariableOpReadVariableOp3batch_normalization_36_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_36/Cast/ReadVariableOpÎ
,batch_normalization_36/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_36_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_36/Cast_1/ReadVariableOpÎ
,batch_normalization_36/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_36_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_36/Cast_2/ReadVariableOpÎ
,batch_normalization_36/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_36_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_36/Cast_3/ReadVariableOp
&batch_normalization_36/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_36/batchnorm/add/yá
$batch_normalization_36/batchnorm/addAddV24batch_normalization_36/Cast_1/ReadVariableOp:value:0/batch_normalization_36/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_36/batchnorm/add¨
&batch_normalization_36/batchnorm/RsqrtRsqrt(batch_normalization_36/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_36/batchnorm/RsqrtÚ
$batch_normalization_36/batchnorm/mulMul*batch_normalization_36/batchnorm/Rsqrt:y:04batch_normalization_36/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_36/batchnorm/mul¶
&batch_normalization_36/batchnorm/mul_1Mulx(batch_normalization_36/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_36/batchnorm/mul_1Ú
&batch_normalization_36/batchnorm/mul_2Mul2batch_normalization_36/Cast/ReadVariableOp:value:0(batch_normalization_36/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_36/batchnorm/mul_2Ú
$batch_normalization_36/batchnorm/subSub4batch_normalization_36/Cast_2/ReadVariableOp:value:0*batch_normalization_36/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_36/batchnorm/subá
&batch_normalization_36/batchnorm/add_1AddV2*batch_normalization_36/batchnorm/mul_1:z:0(batch_normalization_36/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_36/batchnorm/add_1¨
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_30/MatMul/ReadVariableOp²
dense_30/MatMulMatMul*batch_normalization_36/batchnorm/add_1:z:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_30/MatMulÈ
*batch_normalization_37/Cast/ReadVariableOpReadVariableOp3batch_normalization_37_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_37/Cast/ReadVariableOpÎ
,batch_normalization_37/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_37_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_37/Cast_1/ReadVariableOpÎ
,batch_normalization_37/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_37_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_37/Cast_2/ReadVariableOpÎ
,batch_normalization_37/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_37_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_37/Cast_3/ReadVariableOp
&batch_normalization_37/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_37/batchnorm/add/yá
$batch_normalization_37/batchnorm/addAddV24batch_normalization_37/Cast_1/ReadVariableOp:value:0/batch_normalization_37/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/add¨
&batch_normalization_37/batchnorm/RsqrtRsqrt(batch_normalization_37/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_37/batchnorm/RsqrtÚ
$batch_normalization_37/batchnorm/mulMul*batch_normalization_37/batchnorm/Rsqrt:y:04batch_normalization_37/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/mulÎ
&batch_normalization_37/batchnorm/mul_1Muldense_30/MatMul:product:0(batch_normalization_37/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_37/batchnorm/mul_1Ú
&batch_normalization_37/batchnorm/mul_2Mul2batch_normalization_37/Cast/ReadVariableOp:value:0(batch_normalization_37/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_37/batchnorm/mul_2Ú
$batch_normalization_37/batchnorm/subSub4batch_normalization_37/Cast_2/ReadVariableOp:value:0*batch_normalization_37/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_37/batchnorm/subá
&batch_normalization_37/batchnorm/add_1AddV2*batch_normalization_37/batchnorm/mul_1:z:0(batch_normalization_37/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_37/batchnorm/add_1r
ReluRelu*batch_normalization_37/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¨
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_31/MatMul/ReadVariableOp
dense_31/MatMulMatMulRelu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/MatMulÈ
*batch_normalization_38/Cast/ReadVariableOpReadVariableOp3batch_normalization_38_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_38/Cast/ReadVariableOpÎ
,batch_normalization_38/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_38_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_38/Cast_1/ReadVariableOpÎ
,batch_normalization_38/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_38_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_38/Cast_2/ReadVariableOpÎ
,batch_normalization_38/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_38_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_38/Cast_3/ReadVariableOp
&batch_normalization_38/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_38/batchnorm/add/yá
$batch_normalization_38/batchnorm/addAddV24batch_normalization_38/Cast_1/ReadVariableOp:value:0/batch_normalization_38/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/add¨
&batch_normalization_38/batchnorm/RsqrtRsqrt(batch_normalization_38/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_38/batchnorm/RsqrtÚ
$batch_normalization_38/batchnorm/mulMul*batch_normalization_38/batchnorm/Rsqrt:y:04batch_normalization_38/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/mulÎ
&batch_normalization_38/batchnorm/mul_1Muldense_31/MatMul:product:0(batch_normalization_38/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_38/batchnorm/mul_1Ú
&batch_normalization_38/batchnorm/mul_2Mul2batch_normalization_38/Cast/ReadVariableOp:value:0(batch_normalization_38/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_38/batchnorm/mul_2Ú
$batch_normalization_38/batchnorm/subSub4batch_normalization_38/Cast_2/ReadVariableOp:value:0*batch_normalization_38/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_38/batchnorm/subá
&batch_normalization_38/batchnorm/add_1AddV2*batch_normalization_38/batchnorm/mul_1:z:0(batch_normalization_38/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_38/batchnorm/add_1v
Relu_1Relu*batch_normalization_38/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1¨
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_32/MatMul/ReadVariableOp
dense_32/MatMulMatMulRelu_1:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_32/MatMulÈ
*batch_normalization_39/Cast/ReadVariableOpReadVariableOp3batch_normalization_39_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_39/Cast/ReadVariableOpÎ
,batch_normalization_39/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_39_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_39/Cast_1/ReadVariableOpÎ
,batch_normalization_39/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_39_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_39/Cast_2/ReadVariableOpÎ
,batch_normalization_39/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_39_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_39/Cast_3/ReadVariableOp
&batch_normalization_39/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_39/batchnorm/add/yá
$batch_normalization_39/batchnorm/addAddV24batch_normalization_39/Cast_1/ReadVariableOp:value:0/batch_normalization_39/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/add¨
&batch_normalization_39/batchnorm/RsqrtRsqrt(batch_normalization_39/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_39/batchnorm/RsqrtÚ
$batch_normalization_39/batchnorm/mulMul*batch_normalization_39/batchnorm/Rsqrt:y:04batch_normalization_39/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/mulÎ
&batch_normalization_39/batchnorm/mul_1Muldense_32/MatMul:product:0(batch_normalization_39/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_39/batchnorm/mul_1Ú
&batch_normalization_39/batchnorm/mul_2Mul2batch_normalization_39/Cast/ReadVariableOp:value:0(batch_normalization_39/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_39/batchnorm/mul_2Ú
$batch_normalization_39/batchnorm/subSub4batch_normalization_39/Cast_2/ReadVariableOp:value:0*batch_normalization_39/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_39/batchnorm/subá
&batch_normalization_39/batchnorm/add_1AddV2*batch_normalization_39/batchnorm/mul_1:z:0(batch_normalization_39/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_39/batchnorm/add_1v
Relu_2Relu*batch_normalization_39/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¨
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_33/MatMul/ReadVariableOp
dense_33/MatMulMatMulRelu_2:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_33/MatMulÈ
*batch_normalization_40/Cast/ReadVariableOpReadVariableOp3batch_normalization_40_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_40/Cast/ReadVariableOpÎ
,batch_normalization_40/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_40_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_40/Cast_1/ReadVariableOpÎ
,batch_normalization_40/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_40_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_40/Cast_2/ReadVariableOpÎ
,batch_normalization_40/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_40_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_40/Cast_3/ReadVariableOp
&batch_normalization_40/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_40/batchnorm/add/yá
$batch_normalization_40/batchnorm/addAddV24batch_normalization_40/Cast_1/ReadVariableOp:value:0/batch_normalization_40/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/add¨
&batch_normalization_40/batchnorm/RsqrtRsqrt(batch_normalization_40/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_40/batchnorm/RsqrtÚ
$batch_normalization_40/batchnorm/mulMul*batch_normalization_40/batchnorm/Rsqrt:y:04batch_normalization_40/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/mulÎ
&batch_normalization_40/batchnorm/mul_1Muldense_33/MatMul:product:0(batch_normalization_40/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_40/batchnorm/mul_1Ú
&batch_normalization_40/batchnorm/mul_2Mul2batch_normalization_40/Cast/ReadVariableOp:value:0(batch_normalization_40/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_40/batchnorm/mul_2Ú
$batch_normalization_40/batchnorm/subSub4batch_normalization_40/Cast_2/ReadVariableOp:value:0*batch_normalization_40/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_40/batchnorm/subá
&batch_normalization_40/batchnorm/add_1AddV2*batch_normalization_40/batchnorm/mul_1:z:0(batch_normalization_40/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_40/batchnorm/add_1v
Relu_3Relu*batch_normalization_40/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¨
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_34/MatMul/ReadVariableOp
dense_34/MatMulMatMulRelu_3:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_34/MatMul§
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_34/BiasAdd/ReadVariableOp¥
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_34/BiasAddt
IdentityIdentitydense_34/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity·	
NoOpNoOp+^batch_normalization_36/Cast/ReadVariableOp-^batch_normalization_36/Cast_1/ReadVariableOp-^batch_normalization_36/Cast_2/ReadVariableOp-^batch_normalization_36/Cast_3/ReadVariableOp+^batch_normalization_37/Cast/ReadVariableOp-^batch_normalization_37/Cast_1/ReadVariableOp-^batch_normalization_37/Cast_2/ReadVariableOp-^batch_normalization_37/Cast_3/ReadVariableOp+^batch_normalization_38/Cast/ReadVariableOp-^batch_normalization_38/Cast_1/ReadVariableOp-^batch_normalization_38/Cast_2/ReadVariableOp-^batch_normalization_38/Cast_3/ReadVariableOp+^batch_normalization_39/Cast/ReadVariableOp-^batch_normalization_39/Cast_1/ReadVariableOp-^batch_normalization_39/Cast_2/ReadVariableOp-^batch_normalization_39/Cast_3/ReadVariableOp+^batch_normalization_40/Cast/ReadVariableOp-^batch_normalization_40/Cast_1/ReadVariableOp-^batch_normalization_40/Cast_2/ReadVariableOp-^batch_normalization_40/Cast_3/ReadVariableOp^dense_30/MatMul/ReadVariableOp^dense_31/MatMul/ReadVariableOp^dense_32/MatMul/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_36/Cast/ReadVariableOp*batch_normalization_36/Cast/ReadVariableOp2\
,batch_normalization_36/Cast_1/ReadVariableOp,batch_normalization_36/Cast_1/ReadVariableOp2\
,batch_normalization_36/Cast_2/ReadVariableOp,batch_normalization_36/Cast_2/ReadVariableOp2\
,batch_normalization_36/Cast_3/ReadVariableOp,batch_normalization_36/Cast_3/ReadVariableOp2X
*batch_normalization_37/Cast/ReadVariableOp*batch_normalization_37/Cast/ReadVariableOp2\
,batch_normalization_37/Cast_1/ReadVariableOp,batch_normalization_37/Cast_1/ReadVariableOp2\
,batch_normalization_37/Cast_2/ReadVariableOp,batch_normalization_37/Cast_2/ReadVariableOp2\
,batch_normalization_37/Cast_3/ReadVariableOp,batch_normalization_37/Cast_3/ReadVariableOp2X
*batch_normalization_38/Cast/ReadVariableOp*batch_normalization_38/Cast/ReadVariableOp2\
,batch_normalization_38/Cast_1/ReadVariableOp,batch_normalization_38/Cast_1/ReadVariableOp2\
,batch_normalization_38/Cast_2/ReadVariableOp,batch_normalization_38/Cast_2/ReadVariableOp2\
,batch_normalization_38/Cast_3/ReadVariableOp,batch_normalization_38/Cast_3/ReadVariableOp2X
*batch_normalization_39/Cast/ReadVariableOp*batch_normalization_39/Cast/ReadVariableOp2\
,batch_normalization_39/Cast_1/ReadVariableOp,batch_normalization_39/Cast_1/ReadVariableOp2\
,batch_normalization_39/Cast_2/ReadVariableOp,batch_normalization_39/Cast_2/ReadVariableOp2\
,batch_normalization_39/Cast_3/ReadVariableOp,batch_normalization_39/Cast_3/ReadVariableOp2X
*batch_normalization_40/Cast/ReadVariableOp*batch_normalization_40/Cast/ReadVariableOp2\
,batch_normalization_40/Cast_1/ReadVariableOp,batch_normalization_40/Cast_1/ReadVariableOp2\
,batch_normalization_40/Cast_2/ReadVariableOp,batch_normalization_40/Cast_2/ReadVariableOp2\
,batch_normalization_40/Cast_3/ReadVariableOp,batch_normalization_40/Cast_3/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
±

R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_323849

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¬
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë+
Ó
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_323967

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
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

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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

:*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
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
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp 
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
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
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityæ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¨L
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
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ
<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:«Æ
ö
	bn_layers
dense_layers
	variables
trainable_variables
regularization_losses
	keras_api

signatures
_default_save_signature
+&call_and_return_all_conditional_losses
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
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25"
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
'10
(11
)12
*13
+14
,15"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
-metrics

.layers
	variables
trainable_variables
regularization_losses
/non_trainable_variables
0layer_regularization_losses
1layer_metrics
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
ì
2axis
	gamma
beta
moving_mean
moving_variance
3	variables
4trainable_variables
5regularization_losses
6	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
7axis
	gamma
beta
moving_mean
 moving_variance
8	variables
9trainable_variables
:regularization_losses
;	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
<axis
	gamma
beta
!moving_mean
"moving_variance
=	variables
>trainable_variables
?regularization_losses
@	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
Aaxis
	gamma
beta
#moving_mean
$moving_variance
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
Faxis
	gamma
beta
%moving_mean
&moving_variance
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
³

'kernel
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
+ &call_and_return_all_conditional_losses
¡__call__"
_tf_keras_layer
³

(kernel
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"
_tf_keras_layer
³

)kernel
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
+¤&call_and_return_all_conditional_losses
¥__call__"
_tf_keras_layer
³

*kernel
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
+¦&call_and_return_all_conditional_losses
§__call__"
_tf_keras_layer
½

+kernel
,bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
+¨&call_and_return_all_conditional_losses
©__call__"
_tf_keras_layer
A:?
23feed_forward_sub_net_6/batch_normalization_36/gamma
@:>
22feed_forward_sub_net_6/batch_normalization_36/beta
A:?23feed_forward_sub_net_6/batch_normalization_37/gamma
@:>22feed_forward_sub_net_6/batch_normalization_37/beta
A:?23feed_forward_sub_net_6/batch_normalization_38/gamma
@:>22feed_forward_sub_net_6/batch_normalization_38/beta
A:?23feed_forward_sub_net_6/batch_normalization_39/gamma
@:>22feed_forward_sub_net_6/batch_normalization_39/beta
A:?23feed_forward_sub_net_6/batch_normalization_40/gamma
@:>22feed_forward_sub_net_6/batch_normalization_40/beta
I:G
 (29feed_forward_sub_net_6/batch_normalization_36/moving_mean
M:K
 (2=feed_forward_sub_net_6/batch_normalization_36/moving_variance
I:G (29feed_forward_sub_net_6/batch_normalization_37/moving_mean
M:K (2=feed_forward_sub_net_6/batch_normalization_37/moving_variance
I:G (29feed_forward_sub_net_6/batch_normalization_38/moving_mean
M:K (2=feed_forward_sub_net_6/batch_normalization_38/moving_variance
I:G (29feed_forward_sub_net_6/batch_normalization_39/moving_mean
M:K (2=feed_forward_sub_net_6/batch_normalization_39/moving_variance
I:G (29feed_forward_sub_net_6/batch_normalization_40/moving_mean
M:K (2=feed_forward_sub_net_6/batch_normalization_40/moving_variance
8:6
2&feed_forward_sub_net_6/dense_30/kernel
8:62&feed_forward_sub_net_6/dense_31/kernel
8:62&feed_forward_sub_net_6/dense_32/kernel
8:62&feed_forward_sub_net_6/dense_33/kernel
8:6
2&feed_forward_sub_net_6/dense_34/kernel
2:0
2$feed_forward_sub_net_6/dense_34/bias
 "
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
f
0
1
2
 3
!4
"5
#6
$7
%8
&9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
`metrics

alayers
3	variables
4trainable_variables
5regularization_losses
bnon_trainable_variables
clayer_regularization_losses
dlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
emetrics

flayers
8	variables
9trainable_variables
:regularization_losses
gnon_trainable_variables
hlayer_regularization_losses
ilayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
!2
"3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
jmetrics

klayers
=	variables
>trainable_variables
?regularization_losses
lnon_trainable_variables
mlayer_regularization_losses
nlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
#2
$3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
ometrics

players
B	variables
Ctrainable_variables
Dregularization_losses
qnon_trainable_variables
rlayer_regularization_losses
slayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
%2
&3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
tmetrics

ulayers
G	variables
Htrainable_variables
Iregularization_losses
vnon_trainable_variables
wlayer_regularization_losses
xlayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
'
'0"
trackable_list_wrapper
'
'0"
trackable_list_wrapper
 "
trackable_list_wrapper
°
ymetrics

zlayers
L	variables
Mtrainable_variables
Nregularization_losses
{non_trainable_variables
|layer_regularization_losses
}layer_metrics
¡__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
'
(0"
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
³
~metrics

layers
P	variables
Qtrainable_variables
Rregularization_losses
non_trainable_variables
 layer_regularization_losses
layer_metrics
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
'
)0"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
layers
T	variables
Utrainable_variables
Vregularization_losses
non_trainable_variables
 layer_regularization_losses
layer_metrics
¥__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
layers
X	variables
Ytrainable_variables
Zregularization_losses
non_trainable_variables
 layer_regularization_losses
layer_metrics
§__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
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
µ
metrics
layers
\	variables
]trainable_variables
^regularization_losses
non_trainable_variables
 layer_regularization_losses
layer_metrics
©__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
0
 1"
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
!0
"1"
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
#0
$1"
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
%0
&1"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÌBÉ
!__inference__wrapped_model_321451input_1"
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
2þ
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_323041
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_323227
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_323333
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_323519«
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
2
7__inference_feed_forward_sub_net_6_layer_call_fn_323576
7__inference_feed_forward_sub_net_6_layer_call_fn_323633
7__inference_feed_forward_sub_net_6_layer_call_fn_323690
7__inference_feed_forward_sub_net_6_layer_call_fn_323747«
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
$__inference_signature_wrapper_322935input_1"
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
â2ß
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_323767
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_323803´
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
7__inference_batch_normalization_36_layer_call_fn_323816
7__inference_batch_normalization_36_layer_call_fn_323829´
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
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_323849
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_323885´
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
7__inference_batch_normalization_37_layer_call_fn_323898
7__inference_batch_normalization_37_layer_call_fn_323911´
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
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_323931
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_323967´
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
7__inference_batch_normalization_38_layer_call_fn_323980
7__inference_batch_normalization_38_layer_call_fn_323993´
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
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_324013
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_324049´
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
7__inference_batch_normalization_39_layer_call_fn_324062
7__inference_batch_normalization_39_layer_call_fn_324075´
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
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_324095
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_324131´
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
7__inference_batch_normalization_40_layer_call_fn_324144
7__inference_batch_normalization_40_layer_call_fn_324157´
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
î2ë
D__inference_dense_30_layer_call_and_return_conditional_losses_324164¢
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
Ó2Ð
)__inference_dense_30_layer_call_fn_324171¢
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
î2ë
D__inference_dense_31_layer_call_and_return_conditional_losses_324178¢
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
Ó2Ð
)__inference_dense_31_layer_call_fn_324185¢
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
î2ë
D__inference_dense_32_layer_call_and_return_conditional_losses_324192¢
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
Ó2Ð
)__inference_dense_32_layer_call_fn_324199¢
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
î2ë
D__inference_dense_33_layer_call_and_return_conditional_losses_324206¢
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
Ó2Ð
)__inference_dense_33_layer_call_fn_324213¢
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
î2ë
D__inference_dense_34_layer_call_and_return_conditional_losses_324223¢
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
Ó2Ð
)__inference_dense_34_layer_call_fn_324232¢
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
!__inference__wrapped_model_321451' (!")#$*%&+,0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
¸
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_323767b3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¸
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_323803b3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
7__inference_batch_normalization_36_layer_call_fn_323816U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ

7__inference_batch_normalization_36_layer_call_fn_323829U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ
¸
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_323849b 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_323885b 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_37_layer_call_fn_323898U 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_37_layer_call_fn_323911U 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¸
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_323931b!"3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_323967b!"3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_38_layer_call_fn_323980U!"3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_38_layer_call_fn_323993U!"3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¸
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_324013b#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
R__inference_batch_normalization_39_layer_call_and_return_conditional_losses_324049b#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_39_layer_call_fn_324062U#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_39_layer_call_fn_324075U#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¸
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_324095b%&3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
R__inference_batch_normalization_40_layer_call_and_return_conditional_losses_324131b%&3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_40_layer_call_fn_324144U%&3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_40_layer_call_fn_324157U%&3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ£
D__inference_dense_30_layer_call_and_return_conditional_losses_324164['/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
)__inference_dense_30_layer_call_fn_324171N'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ£
D__inference_dense_31_layer_call_and_return_conditional_losses_324178[(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
)__inference_dense_31_layer_call_fn_324185N(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
D__inference_dense_32_layer_call_and_return_conditional_losses_324192[)/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
)__inference_dense_32_layer_call_fn_324199N)/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
D__inference_dense_33_layer_call_and_return_conditional_losses_324206[*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
)__inference_dense_33_layer_call_fn_324213N*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_34_layer_call_and_return_conditional_losses_324223\+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 |
)__inference_dense_34_layer_call_fn_324232O+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
É
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_323041s' (!")#$*%&+,.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 É
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_323227s' (!")#$*%&+,.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ï
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_323333y' (!")#$*%&+,4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ï
R__inference_feed_forward_sub_net_6_layer_call_and_return_conditional_losses_323519y' (!")#$*%&+,4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 §
7__inference_feed_forward_sub_net_6_layer_call_fn_323576l' (!")#$*%&+,4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
¡
7__inference_feed_forward_sub_net_6_layer_call_fn_323633f' (!")#$*%&+,.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
¡
7__inference_feed_forward_sub_net_6_layer_call_fn_323690f' (!")#$*%&+,.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ
§
7__inference_feed_forward_sub_net_6_layer_call_fn_323747l' (!")#$*%&+,4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ
·
$__inference_signature_wrapper_322935' (!")#$*%&+,;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ
"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
