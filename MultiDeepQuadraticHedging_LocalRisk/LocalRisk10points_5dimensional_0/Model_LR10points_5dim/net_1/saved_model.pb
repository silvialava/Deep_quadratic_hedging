��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
executor_typestring �
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.02unknown8ހ
�
2feed_forward_sub_net_1/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*C
shared_name42feed_forward_sub_net_1/batch_normalization_6/gamma
�
Ffeed_forward_sub_net_1/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_1/batch_normalization_6/gamma*
_output_shapes
:
*
dtype0
�
1feed_forward_sub_net_1/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*B
shared_name31feed_forward_sub_net_1/batch_normalization_6/beta
�
Efeed_forward_sub_net_1/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp1feed_forward_sub_net_1/batch_normalization_6/beta*
_output_shapes
:
*
dtype0
�
2feed_forward_sub_net_1/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_1/batch_normalization_7/gamma
�
Ffeed_forward_sub_net_1/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_1/batch_normalization_7/gamma*
_output_shapes
:*
dtype0
�
1feed_forward_sub_net_1/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31feed_forward_sub_net_1/batch_normalization_7/beta
�
Efeed_forward_sub_net_1/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp1feed_forward_sub_net_1/batch_normalization_7/beta*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_1/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_1/batch_normalization_8/gamma
�
Ffeed_forward_sub_net_1/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_1/batch_normalization_8/gamma*
_output_shapes
:*
dtype0
�
1feed_forward_sub_net_1/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31feed_forward_sub_net_1/batch_normalization_8/beta
�
Efeed_forward_sub_net_1/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp1feed_forward_sub_net_1/batch_normalization_8/beta*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_1/batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_1/batch_normalization_9/gamma
�
Ffeed_forward_sub_net_1/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_1/batch_normalization_9/gamma*
_output_shapes
:*
dtype0
�
1feed_forward_sub_net_1/batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31feed_forward_sub_net_1/batch_normalization_9/beta
�
Efeed_forward_sub_net_1/batch_normalization_9/beta/Read/ReadVariableOpReadVariableOp1feed_forward_sub_net_1/batch_normalization_9/beta*
_output_shapes
:*
dtype0
�
3feed_forward_sub_net_1/batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_1/batch_normalization_10/gamma
�
Gfeed_forward_sub_net_1/batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_1/batch_normalization_10/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_1/batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_1/batch_normalization_10/beta
�
Ffeed_forward_sub_net_1/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_1/batch_normalization_10/beta*
_output_shapes
:*
dtype0
�
8feed_forward_sub_net_1/batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*I
shared_name:8feed_forward_sub_net_1/batch_normalization_6/moving_mean
�
Lfeed_forward_sub_net_1/batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp8feed_forward_sub_net_1/batch_normalization_6/moving_mean*
_output_shapes
:
*
dtype0
�
<feed_forward_sub_net_1/batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*M
shared_name><feed_forward_sub_net_1/batch_normalization_6/moving_variance
�
Pfeed_forward_sub_net_1/batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp<feed_forward_sub_net_1/batch_normalization_6/moving_variance*
_output_shapes
:
*
dtype0
�
8feed_forward_sub_net_1/batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8feed_forward_sub_net_1/batch_normalization_7/moving_mean
�
Lfeed_forward_sub_net_1/batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp8feed_forward_sub_net_1/batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
�
<feed_forward_sub_net_1/batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><feed_forward_sub_net_1/batch_normalization_7/moving_variance
�
Pfeed_forward_sub_net_1/batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp<feed_forward_sub_net_1/batch_normalization_7/moving_variance*
_output_shapes
:*
dtype0
�
8feed_forward_sub_net_1/batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8feed_forward_sub_net_1/batch_normalization_8/moving_mean
�
Lfeed_forward_sub_net_1/batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp8feed_forward_sub_net_1/batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
�
<feed_forward_sub_net_1/batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><feed_forward_sub_net_1/batch_normalization_8/moving_variance
�
Pfeed_forward_sub_net_1/batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp<feed_forward_sub_net_1/batch_normalization_8/moving_variance*
_output_shapes
:*
dtype0
�
8feed_forward_sub_net_1/batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8feed_forward_sub_net_1/batch_normalization_9/moving_mean
�
Lfeed_forward_sub_net_1/batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp8feed_forward_sub_net_1/batch_normalization_9/moving_mean*
_output_shapes
:*
dtype0
�
<feed_forward_sub_net_1/batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><feed_forward_sub_net_1/batch_normalization_9/moving_variance
�
Pfeed_forward_sub_net_1/batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp<feed_forward_sub_net_1/batch_normalization_9/moving_variance*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_1/batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_1/batch_normalization_10/moving_mean
�
Mfeed_forward_sub_net_1/batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_1/batch_normalization_10/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_1/batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_1/batch_normalization_10/moving_variance
�
Qfeed_forward_sub_net_1/batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_1/batch_normalization_10/moving_variance*
_output_shapes
:*
dtype0
�
%feed_forward_sub_net_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*6
shared_name'%feed_forward_sub_net_1/dense_5/kernel
�
9feed_forward_sub_net_1/dense_5/kernel/Read/ReadVariableOpReadVariableOp%feed_forward_sub_net_1/dense_5/kernel*
_output_shapes

:
*
dtype0
�
%feed_forward_sub_net_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%feed_forward_sub_net_1/dense_6/kernel
�
9feed_forward_sub_net_1/dense_6/kernel/Read/ReadVariableOpReadVariableOp%feed_forward_sub_net_1/dense_6/kernel*
_output_shapes

:*
dtype0
�
%feed_forward_sub_net_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%feed_forward_sub_net_1/dense_7/kernel
�
9feed_forward_sub_net_1/dense_7/kernel/Read/ReadVariableOpReadVariableOp%feed_forward_sub_net_1/dense_7/kernel*
_output_shapes

:*
dtype0
�
%feed_forward_sub_net_1/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%feed_forward_sub_net_1/dense_8/kernel
�
9feed_forward_sub_net_1/dense_8/kernel/Read/ReadVariableOpReadVariableOp%feed_forward_sub_net_1/dense_8/kernel*
_output_shapes

:*
dtype0
�
%feed_forward_sub_net_1/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*6
shared_name'%feed_forward_sub_net_1/dense_9/kernel
�
9feed_forward_sub_net_1/dense_9/kernel/Read/ReadVariableOpReadVariableOp%feed_forward_sub_net_1/dense_9/kernel*
_output_shapes

:
*
dtype0
�
#feed_forward_sub_net_1/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#feed_forward_sub_net_1/dense_9/bias
�
7feed_forward_sub_net_1/dense_9/bias/Read/ReadVariableOpReadVariableOp#feed_forward_sub_net_1/dense_9/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
�:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�9
value�9B�9 B�9
�
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
�
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
�
-metrics

.layers
	variables
trainable_variables
regularization_losses
/non_trainable_variables
0layer_regularization_losses
1layer_metrics
 
�
2axis
	gamma
beta
moving_mean
moving_variance
3	variables
4trainable_variables
5regularization_losses
6	keras_api
�
7axis
	gamma
beta
moving_mean
 moving_variance
8	variables
9trainable_variables
:regularization_losses
;	keras_api
�
<axis
	gamma
beta
!moving_mean
"moving_variance
=	variables
>trainable_variables
?regularization_losses
@	keras_api
�
Aaxis
	gamma
beta
#moving_mean
$moving_variance
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
�
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
nl
VARIABLE_VALUE2feed_forward_sub_net_1/batch_normalization_6/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1feed_forward_sub_net_1/batch_normalization_6/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_1/batch_normalization_7/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1feed_forward_sub_net_1/batch_normalization_7/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_1/batch_normalization_8/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1feed_forward_sub_net_1/batch_normalization_8/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_1/batch_normalization_9/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1feed_forward_sub_net_1/batch_normalization_9/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_1/batch_normalization_10/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_1/batch_normalization_10/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
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
b`
VARIABLE_VALUE%feed_forward_sub_net_1/dense_5/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%feed_forward_sub_net_1/dense_6/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%feed_forward_sub_net_1/dense_7/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%feed_forward_sub_net_1/dense_8/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%feed_forward_sub_net_1/dense_9/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#feed_forward_sub_net_1/dense_9/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
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
�
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
�
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
�
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
�
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
�
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
�
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
�
~metrics

layers
P	variables
Qtrainable_variables
Rregularization_losses
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics

)0

)0
 
�
�metrics
�layers
T	variables
Utrainable_variables
Vregularization_losses
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics

*0

*0
 
�
�metrics
�layers
X	variables
Ytrainable_variables
Zregularization_losses
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics

+0
,1

+0
,1
 
�
�metrics
�layers
\	variables
]trainable_variables
^regularization_losses
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
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
:���������
*
dtype0*
shape:���������

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_18feed_forward_sub_net_1/batch_normalization_6/moving_mean<feed_forward_sub_net_1/batch_normalization_6/moving_variance1feed_forward_sub_net_1/batch_normalization_6/beta2feed_forward_sub_net_1/batch_normalization_6/gamma%feed_forward_sub_net_1/dense_5/kernel8feed_forward_sub_net_1/batch_normalization_7/moving_mean<feed_forward_sub_net_1/batch_normalization_7/moving_variance1feed_forward_sub_net_1/batch_normalization_7/beta2feed_forward_sub_net_1/batch_normalization_7/gamma%feed_forward_sub_net_1/dense_6/kernel8feed_forward_sub_net_1/batch_normalization_8/moving_mean<feed_forward_sub_net_1/batch_normalization_8/moving_variance1feed_forward_sub_net_1/batch_normalization_8/beta2feed_forward_sub_net_1/batch_normalization_8/gamma%feed_forward_sub_net_1/dense_7/kernel8feed_forward_sub_net_1/batch_normalization_9/moving_mean<feed_forward_sub_net_1/batch_normalization_9/moving_variance1feed_forward_sub_net_1/batch_normalization_9/beta2feed_forward_sub_net_1/batch_normalization_9/gamma%feed_forward_sub_net_1/dense_8/kernel9feed_forward_sub_net_1/batch_normalization_10/moving_mean=feed_forward_sub_net_1/batch_normalization_10/moving_variance2feed_forward_sub_net_1/batch_normalization_10/beta3feed_forward_sub_net_1/batch_normalization_10/gamma%feed_forward_sub_net_1/dense_9/kernel#feed_forward_sub_net_1/dense_9/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_306955
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameFfeed_forward_sub_net_1/batch_normalization_6/gamma/Read/ReadVariableOpEfeed_forward_sub_net_1/batch_normalization_6/beta/Read/ReadVariableOpFfeed_forward_sub_net_1/batch_normalization_7/gamma/Read/ReadVariableOpEfeed_forward_sub_net_1/batch_normalization_7/beta/Read/ReadVariableOpFfeed_forward_sub_net_1/batch_normalization_8/gamma/Read/ReadVariableOpEfeed_forward_sub_net_1/batch_normalization_8/beta/Read/ReadVariableOpFfeed_forward_sub_net_1/batch_normalization_9/gamma/Read/ReadVariableOpEfeed_forward_sub_net_1/batch_normalization_9/beta/Read/ReadVariableOpGfeed_forward_sub_net_1/batch_normalization_10/gamma/Read/ReadVariableOpFfeed_forward_sub_net_1/batch_normalization_10/beta/Read/ReadVariableOpLfeed_forward_sub_net_1/batch_normalization_6/moving_mean/Read/ReadVariableOpPfeed_forward_sub_net_1/batch_normalization_6/moving_variance/Read/ReadVariableOpLfeed_forward_sub_net_1/batch_normalization_7/moving_mean/Read/ReadVariableOpPfeed_forward_sub_net_1/batch_normalization_7/moving_variance/Read/ReadVariableOpLfeed_forward_sub_net_1/batch_normalization_8/moving_mean/Read/ReadVariableOpPfeed_forward_sub_net_1/batch_normalization_8/moving_variance/Read/ReadVariableOpLfeed_forward_sub_net_1/batch_normalization_9/moving_mean/Read/ReadVariableOpPfeed_forward_sub_net_1/batch_normalization_9/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_1/batch_normalization_10/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_1/batch_normalization_10/moving_variance/Read/ReadVariableOp9feed_forward_sub_net_1/dense_5/kernel/Read/ReadVariableOp9feed_forward_sub_net_1/dense_6/kernel/Read/ReadVariableOp9feed_forward_sub_net_1/dense_7/kernel/Read/ReadVariableOp9feed_forward_sub_net_1/dense_8/kernel/Read/ReadVariableOp9feed_forward_sub_net_1/dense_9/kernel/Read/ReadVariableOp7feed_forward_sub_net_1/dense_9/bias/Read/ReadVariableOpConst*'
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
GPU 2J 8� *(
f#R!
__inference__traced_save_308353
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename2feed_forward_sub_net_1/batch_normalization_6/gamma1feed_forward_sub_net_1/batch_normalization_6/beta2feed_forward_sub_net_1/batch_normalization_7/gamma1feed_forward_sub_net_1/batch_normalization_7/beta2feed_forward_sub_net_1/batch_normalization_8/gamma1feed_forward_sub_net_1/batch_normalization_8/beta2feed_forward_sub_net_1/batch_normalization_9/gamma1feed_forward_sub_net_1/batch_normalization_9/beta3feed_forward_sub_net_1/batch_normalization_10/gamma2feed_forward_sub_net_1/batch_normalization_10/beta8feed_forward_sub_net_1/batch_normalization_6/moving_mean<feed_forward_sub_net_1/batch_normalization_6/moving_variance8feed_forward_sub_net_1/batch_normalization_7/moving_mean<feed_forward_sub_net_1/batch_normalization_7/moving_variance8feed_forward_sub_net_1/batch_normalization_8/moving_mean<feed_forward_sub_net_1/batch_normalization_8/moving_variance8feed_forward_sub_net_1/batch_normalization_9/moving_mean<feed_forward_sub_net_1/batch_normalization_9/moving_variance9feed_forward_sub_net_1/batch_normalization_10/moving_mean=feed_forward_sub_net_1/batch_normalization_10/moving_variance%feed_forward_sub_net_1/dense_5/kernel%feed_forward_sub_net_1/dense_6/kernel%feed_forward_sub_net_1/dense_7/kernel%feed_forward_sub_net_1/dense_8/kernel%feed_forward_sub_net_1/dense_9/kernel#feed_forward_sub_net_1/dense_9/bias*&
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_308441��
�
�
6__inference_batch_normalization_6_layer_call_fn_307836

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3054952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
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
:���������
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�+
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_305723

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
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
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_9_layer_call_fn_308095

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3060552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_8_layer_call_and_return_conditional_losses_306387

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_6_layer_call_and_return_conditional_losses_306345

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_7_layer_call_fn_307918

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3056612
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�C
�
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_306418
x*
batch_normalization_6_306308:
*
batch_normalization_6_306310:
*
batch_normalization_6_306312:
*
batch_normalization_6_306314:
 
dense_5_306325:
*
batch_normalization_7_306328:*
batch_normalization_7_306330:*
batch_normalization_7_306332:*
batch_normalization_7_306334: 
dense_6_306346:*
batch_normalization_8_306349:*
batch_normalization_8_306351:*
batch_normalization_8_306353:*
batch_normalization_8_306355: 
dense_7_306367:*
batch_normalization_9_306370:*
batch_normalization_9_306372:*
batch_normalization_9_306374:*
batch_normalization_9_306376: 
dense_8_306388:+
batch_normalization_10_306391:+
batch_normalization_10_306393:+
batch_normalization_10_306395:+
batch_normalization_10_306397: 
dense_9_306412:

dense_9_306414:

identity��.batch_normalization_10/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_6_306308batch_normalization_6_306310batch_normalization_6_306312batch_normalization_6_306314*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3054952/
-batch_normalization_6/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_5_306325*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3063242!
dense_5/StatefulPartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_7_306328batch_normalization_7_306330batch_normalization_7_306332batch_normalization_7_306334*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3056612/
-batch_normalization_7/StatefulPartitionedCall~
ReluRelu6batch_normalization_7/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu�
dense_6/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_6_306346*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3063452!
dense_6/StatefulPartitionedCall�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_8_306349batch_normalization_8_306351batch_normalization_8_306353batch_normalization_8_306355*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_3058272/
-batch_normalization_8/StatefulPartitionedCall�
Relu_1Relu6batch_normalization_8/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_7/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_7_306367*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3063662!
dense_7/StatefulPartitionedCall�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_9_306370batch_normalization_9_306372batch_normalization_9_306374batch_normalization_9_306376*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3059932/
-batch_normalization_9/StatefulPartitionedCall�
Relu_2Relu6batch_normalization_9/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_8/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_8_306388*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3063872!
dense_8/StatefulPartitionedCall�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_10_306391batch_normalization_10_306393batch_normalization_10_306395batch_normalization_10_306397*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_30615920
.batch_normalization_10/StatefulPartitionedCall�
Relu_3Relu7batch_normalization_10/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_9/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_9_306412dense_9_306414*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3064112!
dense_9/StatefulPartitionedCall�
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
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
:���������


_user_specified_namex
�E
�
__inference__traced_save_308353
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
Msavev2_feed_forward_sub_net_1_batch_normalization_10_beta_read_readvariableopW
Ssavev2_feed_forward_sub_net_1_batch_normalization_6_moving_mean_read_readvariableop[
Wsavev2_feed_forward_sub_net_1_batch_normalization_6_moving_variance_read_readvariableopW
Ssavev2_feed_forward_sub_net_1_batch_normalization_7_moving_mean_read_readvariableop[
Wsavev2_feed_forward_sub_net_1_batch_normalization_7_moving_variance_read_readvariableopW
Ssavev2_feed_forward_sub_net_1_batch_normalization_8_moving_mean_read_readvariableop[
Wsavev2_feed_forward_sub_net_1_batch_normalization_8_moving_variance_read_readvariableopW
Ssavev2_feed_forward_sub_net_1_batch_normalization_9_moving_mean_read_readvariableop[
Wsavev2_feed_forward_sub_net_1_batch_normalization_9_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_1_batch_normalization_10_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_1_batch_normalization_10_moving_variance_read_readvariableopD
@savev2_feed_forward_sub_net_1_dense_5_kernel_read_readvariableopD
@savev2_feed_forward_sub_net_1_dense_6_kernel_read_readvariableopD
@savev2_feed_forward_sub_net_1_dense_7_kernel_read_readvariableopD
@savev2_feed_forward_sub_net_1_dense_8_kernel_read_readvariableopD
@savev2_feed_forward_sub_net_1_dense_9_kernel_read_readvariableopB
>savev2_feed_forward_sub_net_1_dense_9_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Msavev2_feed_forward_sub_net_1_batch_normalization_6_gamma_read_readvariableopLsavev2_feed_forward_sub_net_1_batch_normalization_6_beta_read_readvariableopMsavev2_feed_forward_sub_net_1_batch_normalization_7_gamma_read_readvariableopLsavev2_feed_forward_sub_net_1_batch_normalization_7_beta_read_readvariableopMsavev2_feed_forward_sub_net_1_batch_normalization_8_gamma_read_readvariableopLsavev2_feed_forward_sub_net_1_batch_normalization_8_beta_read_readvariableopMsavev2_feed_forward_sub_net_1_batch_normalization_9_gamma_read_readvariableopLsavev2_feed_forward_sub_net_1_batch_normalization_9_beta_read_readvariableopNsavev2_feed_forward_sub_net_1_batch_normalization_10_gamma_read_readvariableopMsavev2_feed_forward_sub_net_1_batch_normalization_10_beta_read_readvariableopSsavev2_feed_forward_sub_net_1_batch_normalization_6_moving_mean_read_readvariableopWsavev2_feed_forward_sub_net_1_batch_normalization_6_moving_variance_read_readvariableopSsavev2_feed_forward_sub_net_1_batch_normalization_7_moving_mean_read_readvariableopWsavev2_feed_forward_sub_net_1_batch_normalization_7_moving_variance_read_readvariableopSsavev2_feed_forward_sub_net_1_batch_normalization_8_moving_mean_read_readvariableopWsavev2_feed_forward_sub_net_1_batch_normalization_8_moving_variance_read_readvariableopSsavev2_feed_forward_sub_net_1_batch_normalization_9_moving_mean_read_readvariableopWsavev2_feed_forward_sub_net_1_batch_normalization_9_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_1_batch_normalization_10_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_1_batch_normalization_10_moving_variance_read_readvariableop@savev2_feed_forward_sub_net_1_dense_5_kernel_read_readvariableop@savev2_feed_forward_sub_net_1_dense_6_kernel_read_readvariableop@savev2_feed_forward_sub_net_1_dense_7_kernel_read_readvariableop@savev2_feed_forward_sub_net_1_dense_8_kernel_read_readvariableop@savev2_feed_forward_sub_net_1_dense_9_kernel_read_readvariableop>savev2_feed_forward_sub_net_1_dense_9_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :
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
�
�
7__inference_feed_forward_sub_net_1_layer_call_fn_307767
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
identity��StatefulPartitionedCall�
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
:���������
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_3066442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
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
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
�
�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_305827

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_306159

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
|
(__inference_dense_6_layer_call_fn_308205

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3063452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
|
(__inference_dense_8_layer_call_fn_308233

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3063872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_8_layer_call_fn_308000

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_3058272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_10_layer_call_fn_308164

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3061592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
|
(__inference_dense_7_layer_call_fn_308219

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3063662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_dense_9_layer_call_and_return_conditional_losses_308243

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
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
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_7_layer_call_and_return_conditional_losses_306366

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_307869

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_307247
xK
=batch_normalization_6_assignmovingavg_readvariableop_resource:
M
?batch_normalization_6_assignmovingavg_1_readvariableop_resource:
@
2batch_normalization_6_cast_readvariableop_resource:
B
4batch_normalization_6_cast_1_readvariableop_resource:
8
&dense_5_matmul_readvariableop_resource:
K
=batch_normalization_7_assignmovingavg_readvariableop_resource:M
?batch_normalization_7_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_7_cast_readvariableop_resource:B
4batch_normalization_7_cast_1_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:K
=batch_normalization_8_assignmovingavg_readvariableop_resource:M
?batch_normalization_8_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_8_cast_readvariableop_resource:B
4batch_normalization_8_cast_1_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:K
=batch_normalization_9_assignmovingavg_readvariableop_resource:M
?batch_normalization_9_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_9_cast_readvariableop_resource:B
4batch_normalization_9_cast_1_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:L
>batch_normalization_10_assignmovingavg_readvariableop_resource:N
@batch_normalization_10_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_10_cast_readvariableop_resource:C
5batch_normalization_10_cast_1_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:
5
'dense_9_biasadd_readvariableop_resource:

identity��&batch_normalization_10/AssignMovingAvg�5batch_normalization_10/AssignMovingAvg/ReadVariableOp�(batch_normalization_10/AssignMovingAvg_1�7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_10/Cast/ReadVariableOp�,batch_normalization_10/Cast_1/ReadVariableOp�%batch_normalization_6/AssignMovingAvg�4batch_normalization_6/AssignMovingAvg/ReadVariableOp�'batch_normalization_6/AssignMovingAvg_1�6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�%batch_normalization_7/AssignMovingAvg�4batch_normalization_7/AssignMovingAvg/ReadVariableOp�'batch_normalization_7/AssignMovingAvg_1�6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�%batch_normalization_8/AssignMovingAvg�4batch_normalization_8/AssignMovingAvg/ReadVariableOp�'batch_normalization_8/AssignMovingAvg_1�6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�%batch_normalization_9/AssignMovingAvg�4batch_normalization_9/AssignMovingAvg/ReadVariableOp�'batch_normalization_9/AssignMovingAvg_1�6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_9/Cast/ReadVariableOp�+batch_normalization_9/Cast_1/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/MatMul/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_6/moments/mean/reduction_indices�
"batch_normalization_6/moments/meanMeanx=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2$
"batch_normalization_6/moments/mean�
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes

:
2,
*batch_normalization_6/moments/StopGradient�
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencex3batch_normalization_6/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������
21
/batch_normalization_6/moments/SquaredDifference�
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_6/moments/variance/reduction_indices�
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2(
&batch_normalization_6/moments/variance�
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2'
%batch_normalization_6/moments/Squeeze�
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1�
+batch_normalization_6/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_6/AssignMovingAvg/decay�
*batch_normalization_6/AssignMovingAvg/CastCast4batch_normalization_6/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_6/AssignMovingAvg/Cast�
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOp�
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*
_output_shapes
:
2+
)batch_normalization_6/AssignMovingAvg/sub�
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:0.batch_normalization_6/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2+
)batch_normalization_6/AssignMovingAvg/mul�
%batch_normalization_6/AssignMovingAvgAssignSubVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_6/AssignMovingAvg�
-batch_normalization_6/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_6/AssignMovingAvg_1/decay�
,batch_normalization_6/AssignMovingAvg_1/CastCast6batch_normalization_6/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_6/AssignMovingAvg_1/Cast�
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2-
+batch_normalization_6/AssignMovingAvg_1/sub�
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:00batch_normalization_6/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2-
+batch_normalization_6/AssignMovingAvg_1/mul�
'batch_normalization_6/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_6/AssignMovingAvg_1�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:
*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Mulx'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub1batch_normalization_6/Cast/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2'
%batch_normalization_6/batchnorm/add_1�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/MatMul�
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_7/moments/mean/reduction_indices�
"batch_normalization_7/moments/meanMeandense_5/MatMul:product:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_7/moments/mean�
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_7/moments/StopGradient�
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferencedense_5/MatMul:product:03batch_normalization_7/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������21
/batch_normalization_7/moments/SquaredDifference�
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_7/moments/variance/reduction_indices�
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_7/moments/variance�
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_7/moments/Squeeze�
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1�
+batch_normalization_7/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_7/AssignMovingAvg/decay�
*batch_normalization_7/AssignMovingAvg/CastCast4batch_normalization_7/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_7/AssignMovingAvg/Cast�
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOp�
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/sub�
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:0.batch_normalization_7/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/mul�
%batch_normalization_7/AssignMovingAvgAssignSubVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_7/AssignMovingAvg�
-batch_normalization_7/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_7/AssignMovingAvg_1/decay�
,batch_normalization_7/AssignMovingAvg_1/CastCast6batch_normalization_7/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_7/AssignMovingAvg_1/Cast�
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/sub�
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:00batch_normalization_7/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/mul�
'batch_normalization_7/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_7/AssignMovingAvg_1�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_5/MatMul:product:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub1batch_normalization_7/Cast/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_7/batchnorm/add_1q
ReluRelu)batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp�
dense_6/MatMulMatMulRelu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6/MatMul�
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_8/moments/mean/reduction_indices�
"batch_normalization_8/moments/meanMeandense_6/MatMul:product:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_8/moments/mean�
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_8/moments/StopGradient�
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferencedense_6/MatMul:product:03batch_normalization_8/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������21
/batch_normalization_8/moments/SquaredDifference�
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_8/moments/variance/reduction_indices�
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_8/moments/variance�
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_8/moments/Squeeze�
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1�
+batch_normalization_8/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_8/AssignMovingAvg/decay�
*batch_normalization_8/AssignMovingAvg/CastCast4batch_normalization_8/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_8/AssignMovingAvg/Cast�
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOp�
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_8/AssignMovingAvg/sub�
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:0.batch_normalization_8/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_8/AssignMovingAvg/mul�
%batch_normalization_8/AssignMovingAvgAssignSubVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_8/AssignMovingAvg�
-batch_normalization_8/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_8/AssignMovingAvg_1/decay�
,batch_normalization_8/AssignMovingAvg_1/CastCast6batch_normalization_8/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_8/AssignMovingAvg_1/Cast�
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_8/AssignMovingAvg_1/sub�
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:00batch_normalization_8/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_8/AssignMovingAvg_1/mul�
'batch_normalization_8/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_8/AssignMovingAvg_1�
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp�
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_8/batchnorm/add/y�
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/add�
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/Rsqrt�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/mul�
%batch_normalization_8/batchnorm/mul_1Muldense_6/MatMul:product:0'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_8/batchnorm/mul_1�
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/mul_2�
#batch_normalization_8/batchnorm/subSub1batch_normalization_8/Cast/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/sub�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_8/batchnorm/add_1u
Relu_1Relu)batch_normalization_8/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp�
dense_7/MatMulMatMulRelu_1:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_7/MatMul�
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_9/moments/mean/reduction_indices�
"batch_normalization_9/moments/meanMeandense_7/MatMul:product:0=batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_9/moments/mean�
*batch_normalization_9/moments/StopGradientStopGradient+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_9/moments/StopGradient�
/batch_normalization_9/moments/SquaredDifferenceSquaredDifferencedense_7/MatMul:product:03batch_normalization_9/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������21
/batch_normalization_9/moments/SquaredDifference�
8batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_9/moments/variance/reduction_indices�
&batch_normalization_9/moments/varianceMean3batch_normalization_9/moments/SquaredDifference:z:0Abatch_normalization_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_9/moments/variance�
%batch_normalization_9/moments/SqueezeSqueeze+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_9/moments/Squeeze�
'batch_normalization_9/moments/Squeeze_1Squeeze/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_9/moments/Squeeze_1�
+batch_normalization_9/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_9/AssignMovingAvg/decay�
*batch_normalization_9/AssignMovingAvg/CastCast4batch_normalization_9/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_9/AssignMovingAvg/Cast�
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_9_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOp�
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/sub�
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:0.batch_normalization_9/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/mul�
%batch_normalization_9/AssignMovingAvgAssignSubVariableOp=batch_normalization_9_assignmovingavg_readvariableop_resource-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_9/AssignMovingAvg�
-batch_normalization_9/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_9/AssignMovingAvg_1/decay�
,batch_normalization_9/AssignMovingAvg_1/CastCast6batch_normalization_9/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_9/AssignMovingAvg_1/Cast�
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_9_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/sub�
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:00batch_normalization_9/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/mul�
'batch_normalization_9/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_9_assignmovingavg_1_readvariableop_resource/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_9/AssignMovingAvg_1�
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp�
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV20batch_normalization_9/moments/Squeeze_1:output:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/Rsqrt�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Muldense_7/MatMul:product:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_9/batchnorm/mul_1�
%batch_normalization_9/batchnorm/mul_2Mul.batch_normalization_9/moments/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/mul_2�
#batch_normalization_9/batchnorm/subSub1batch_normalization_9/Cast/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_9/batchnorm/add_1u
Relu_2Relu)batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulRelu_2:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_8/MatMul�
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_10/moments/mean/reduction_indices�
#batch_normalization_10/moments/meanMeandense_8/MatMul:product:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_10/moments/mean�
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_10/moments/StopGradient�
0batch_normalization_10/moments/SquaredDifferenceSquaredDifferencedense_8/MatMul:product:04batch_normalization_10/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_10/moments/SquaredDifference�
9batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_10/moments/variance/reduction_indices�
'batch_normalization_10/moments/varianceMean4batch_normalization_10/moments/SquaredDifference:z:0Bbatch_normalization_10/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_10/moments/variance�
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_10/moments/Squeeze�
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_10/moments/Squeeze_1�
,batch_normalization_10/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_10/AssignMovingAvg/decay�
+batch_normalization_10/AssignMovingAvg/CastCast5batch_normalization_10/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_10/AssignMovingAvg/Cast�
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_10_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOp�
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/sub�
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:0/batch_normalization_10/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/mul�
&batch_normalization_10/AssignMovingAvgAssignSubVariableOp>batch_normalization_10_assignmovingavg_readvariableop_resource.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_10/AssignMovingAvg�
.batch_normalization_10/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_10/AssignMovingAvg_1/decay�
-batch_normalization_10/AssignMovingAvg_1/CastCast7batch_normalization_10/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_10/AssignMovingAvg_1/Cast�
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_10_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/sub�
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:01batch_normalization_10/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/mul�
(batch_normalization_10/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_10_assignmovingavg_1_readvariableop_resource0batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_10/AssignMovingAvg_1�
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp�
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV21batch_normalization_10/moments/Squeeze_1:output:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/Rsqrt�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_8/MatMul:product:0(batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_10/batchnorm/mul_1�
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/mul_2�
$batch_normalization_10/batchnorm/subSub2batch_normalization_10/Cast/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_10/batchnorm/add_1v
Relu_3Relu*batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulRelu_3:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_9/BiasAdds
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp'^batch_normalization_10/AssignMovingAvg6^batch_normalization_10/AssignMovingAvg/ReadVariableOp)^batch_normalization_10/AssignMovingAvg_18^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp&^batch_normalization_6/AssignMovingAvg5^batch_normalization_6/AssignMovingAvg/ReadVariableOp(^batch_normalization_6/AssignMovingAvg_17^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp&^batch_normalization_7/AssignMovingAvg5^batch_normalization_7/AssignMovingAvg/ReadVariableOp(^batch_normalization_7/AssignMovingAvg_17^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp&^batch_normalization_8/AssignMovingAvg5^batch_normalization_8/AssignMovingAvg/ReadVariableOp(^batch_normalization_8/AssignMovingAvg_17^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp&^batch_normalization_9/AssignMovingAvg5^batch_normalization_9/AssignMovingAvg/ReadVariableOp(^batch_normalization_9/AssignMovingAvg_17^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
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
:���������


_user_specified_namex
�+
�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_305889

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
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
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_305661

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�+
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_307905

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
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
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�+
�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307823

inputs5
'assignmovingavg_readvariableop_resource:
7
)assignmovingavg_1_readvariableop_resource:
*
cast_readvariableop_resource:
,
cast_1_readvariableop_resource:

identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
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
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������
2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:
2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast/ReadVariableOp�
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
valueB 2�����ư>2
batchnorm/add/y�
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
:���������
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_7_layer_call_fn_307931

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3057232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_10_layer_call_fn_308177

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3062212
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_5_layer_call_and_return_conditional_losses_306324

inputs0
matmul_readvariableop_resource:

identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������
: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�+
�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_305557

inputs5
'assignmovingavg_readvariableop_resource:
7
)assignmovingavg_1_readvariableop_resource:
*
cast_readvariableop_resource:
,
cast_1_readvariableop_resource:

identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
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
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������
2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:
2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast/ReadVariableOp�
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
valueB 2�����ư>2
batchnorm/add/y�
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
:���������
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_8_layer_call_fn_308013

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_3058892
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�+
�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_307987

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
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
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_306221

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
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
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_305993

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�+
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_308151

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
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
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_feed_forward_sub_net_1_layer_call_fn_307653
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
identity��StatefulPartitionedCall�
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
:���������
*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_3064182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
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
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������


_user_specified_namex
�
�
C__inference_dense_6_layer_call_and_return_conditional_losses_308198

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_6_layer_call_fn_307849

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3055572
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
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
:���������
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�+
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_308069

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
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
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_5_layer_call_and_return_conditional_losses_308184

inputs0
matmul_readvariableop_resource:

identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������
: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�+
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_306055

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
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
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
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
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/CastCastAssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg/Cast�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg/mul�
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
�#<2
AssignMovingAvg_1/decay�
AssignMovingAvg_1/CastCast AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
AssignMovingAvg_1/Cast�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_feed_forward_sub_net_1_layer_call_fn_307710
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
identity��StatefulPartitionedCall�
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
:���������
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_3066442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
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
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������


_user_specified_namex
�
�
(__inference_dense_9_layer_call_fn_308252

inputs
unknown:

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3064112
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
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
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_305471
input_1W
Ifeed_forward_sub_net_1_batch_normalization_6_cast_readvariableop_resource:
Y
Kfeed_forward_sub_net_1_batch_normalization_6_cast_1_readvariableop_resource:
Y
Kfeed_forward_sub_net_1_batch_normalization_6_cast_2_readvariableop_resource:
Y
Kfeed_forward_sub_net_1_batch_normalization_6_cast_3_readvariableop_resource:
O
=feed_forward_sub_net_1_dense_5_matmul_readvariableop_resource:
W
Ifeed_forward_sub_net_1_batch_normalization_7_cast_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_7_cast_1_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_7_cast_2_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_7_cast_3_readvariableop_resource:O
=feed_forward_sub_net_1_dense_6_matmul_readvariableop_resource:W
Ifeed_forward_sub_net_1_batch_normalization_8_cast_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_8_cast_1_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_8_cast_2_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_8_cast_3_readvariableop_resource:O
=feed_forward_sub_net_1_dense_7_matmul_readvariableop_resource:W
Ifeed_forward_sub_net_1_batch_normalization_9_cast_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_9_cast_1_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_9_cast_2_readvariableop_resource:Y
Kfeed_forward_sub_net_1_batch_normalization_9_cast_3_readvariableop_resource:O
=feed_forward_sub_net_1_dense_8_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_1_batch_normalization_10_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_1_batch_normalization_10_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_1_batch_normalization_10_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_1_batch_normalization_10_cast_3_readvariableop_resource:O
=feed_forward_sub_net_1_dense_9_matmul_readvariableop_resource:
L
>feed_forward_sub_net_1_dense_9_biasadd_readvariableop_resource:

identity��Afeed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOp�Cfeed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOp�@feed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOp�Bfeed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOp�Bfeed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOp�Bfeed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOp�@feed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOp�Bfeed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOp�Bfeed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOp�Bfeed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOp�@feed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOp�Bfeed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOp�Bfeed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOp�Bfeed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOp�@feed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOp�Bfeed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOp�Bfeed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOp�Bfeed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOp�4feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOp�4feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOp�4feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOp�4feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOp�5feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOp�4feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOp�
@feed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOpReadVariableOpIfeed_forward_sub_net_1_batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:
*
dtype02B
@feed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOp�
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOp�
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOp�
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOp�
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2>
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/add/y�
:feed_forward_sub_net_1/batch_normalization_6/batchnorm/addAddV2Jfeed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOp:value:0Efeed_forward_sub_net_1/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2<
:feed_forward_sub_net_1/batch_normalization_6/batchnorm/add�
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/RsqrtRsqrt>feed_forward_sub_net_1/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:
2>
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/Rsqrt�
:feed_forward_sub_net_1/batch_normalization_6/batchnorm/mulMul@feed_forward_sub_net_1/batch_normalization_6/batchnorm/Rsqrt:y:0Jfeed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2<
:feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul�
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul_1Mulinput_1>feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2>
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul_1�
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul_2MulHfeed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOp:value:0>feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:
2>
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul_2�
:feed_forward_sub_net_1/batch_normalization_6/batchnorm/subSubJfeed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOp:value:0@feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2<
:feed_forward_sub_net_1/batch_normalization_6/batchnorm/sub�
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/add_1AddV2@feed_forward_sub_net_1/batch_normalization_6/batchnorm/mul_1:z:0>feed_forward_sub_net_1/batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2>
<feed_forward_sub_net_1/batch_normalization_6/batchnorm/add_1�
4feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOpReadVariableOp=feed_forward_sub_net_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype026
4feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOp�
%feed_forward_sub_net_1/dense_5/MatMulMatMul@feed_forward_sub_net_1/batch_normalization_6/batchnorm/add_1:z:0<feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2'
%feed_forward_sub_net_1/dense_5/MatMul�
@feed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOpReadVariableOpIfeed_forward_sub_net_1_batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype02B
@feed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOp�
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOp�
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOp�
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOp�
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2>
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/add/y�
:feed_forward_sub_net_1/batch_normalization_7/batchnorm/addAddV2Jfeed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOp:value:0Efeed_forward_sub_net_1/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_7/batchnorm/add�
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/RsqrtRsqrt>feed_forward_sub_net_1/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/Rsqrt�
:feed_forward_sub_net_1/batch_normalization_7/batchnorm/mulMul@feed_forward_sub_net_1/batch_normalization_7/batchnorm/Rsqrt:y:0Jfeed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul�
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul_1Mul/feed_forward_sub_net_1/dense_5/MatMul:product:0>feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2>
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul_1�
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul_2MulHfeed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOp:value:0>feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul_2�
:feed_forward_sub_net_1/batch_normalization_7/batchnorm/subSubJfeed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOp:value:0@feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_7/batchnorm/sub�
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/add_1AddV2@feed_forward_sub_net_1/batch_normalization_7/batchnorm/mul_1:z:0>feed_forward_sub_net_1/batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2>
<feed_forward_sub_net_1/batch_normalization_7/batchnorm/add_1�
feed_forward_sub_net_1/ReluRelu@feed_forward_sub_net_1/batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_1/Relu�
4feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOpReadVariableOp=feed_forward_sub_net_1_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype026
4feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOp�
%feed_forward_sub_net_1/dense_6/MatMulMatMul)feed_forward_sub_net_1/Relu:activations:0<feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2'
%feed_forward_sub_net_1/dense_6/MatMul�
@feed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOpReadVariableOpIfeed_forward_sub_net_1_batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype02B
@feed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOp�
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOp�
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOp�
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOp�
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2>
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/add/y�
:feed_forward_sub_net_1/batch_normalization_8/batchnorm/addAddV2Jfeed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOp:value:0Efeed_forward_sub_net_1/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_8/batchnorm/add�
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/RsqrtRsqrt>feed_forward_sub_net_1/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/Rsqrt�
:feed_forward_sub_net_1/batch_normalization_8/batchnorm/mulMul@feed_forward_sub_net_1/batch_normalization_8/batchnorm/Rsqrt:y:0Jfeed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul�
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul_1Mul/feed_forward_sub_net_1/dense_6/MatMul:product:0>feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2>
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul_1�
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul_2MulHfeed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOp:value:0>feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul_2�
:feed_forward_sub_net_1/batch_normalization_8/batchnorm/subSubJfeed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOp:value:0@feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_8/batchnorm/sub�
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/add_1AddV2@feed_forward_sub_net_1/batch_normalization_8/batchnorm/mul_1:z:0>feed_forward_sub_net_1/batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2>
<feed_forward_sub_net_1/batch_normalization_8/batchnorm/add_1�
feed_forward_sub_net_1/Relu_1Relu@feed_forward_sub_net_1/batch_normalization_8/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_1/Relu_1�
4feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOpReadVariableOp=feed_forward_sub_net_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype026
4feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOp�
%feed_forward_sub_net_1/dense_7/MatMulMatMul+feed_forward_sub_net_1/Relu_1:activations:0<feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2'
%feed_forward_sub_net_1/dense_7/MatMul�
@feed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOpReadVariableOpIfeed_forward_sub_net_1_batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:*
dtype02B
@feed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOp�
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOp�
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_9_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOp�
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOpReadVariableOpKfeed_forward_sub_net_1_batch_normalization_9_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOp�
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2>
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/add/y�
:feed_forward_sub_net_1/batch_normalization_9/batchnorm/addAddV2Jfeed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOp:value:0Efeed_forward_sub_net_1/batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_9/batchnorm/add�
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/RsqrtRsqrt>feed_forward_sub_net_1/batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/Rsqrt�
:feed_forward_sub_net_1/batch_normalization_9/batchnorm/mulMul@feed_forward_sub_net_1/batch_normalization_9/batchnorm/Rsqrt:y:0Jfeed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul�
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul_1Mul/feed_forward_sub_net_1/dense_7/MatMul:product:0>feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2>
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul_1�
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul_2MulHfeed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOp:value:0>feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2>
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul_2�
:feed_forward_sub_net_1/batch_normalization_9/batchnorm/subSubJfeed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOp:value:0@feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2<
:feed_forward_sub_net_1/batch_normalization_9/batchnorm/sub�
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/add_1AddV2@feed_forward_sub_net_1/batch_normalization_9/batchnorm/mul_1:z:0>feed_forward_sub_net_1/batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2>
<feed_forward_sub_net_1/batch_normalization_9/batchnorm/add_1�
feed_forward_sub_net_1/Relu_2Relu@feed_forward_sub_net_1/batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_1/Relu_2�
4feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOpReadVariableOp=feed_forward_sub_net_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype026
4feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOp�
%feed_forward_sub_net_1/dense_8/MatMulMatMul+feed_forward_sub_net_1/Relu_2:activations:0<feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2'
%feed_forward_sub_net_1/dense_8/MatMul�
Afeed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_1_batch_normalization_10_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOp�
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_1_batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_1_batch_normalization_10_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_1_batch_normalization_10_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOp�
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/add/y�
;feed_forward_sub_net_1/batch_normalization_10/batchnorm/addAddV2Kfeed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_1/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_1/batch_normalization_10/batchnorm/add�
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/RsqrtRsqrt?feed_forward_sub_net_1/batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/Rsqrt�
;feed_forward_sub_net_1/batch_normalization_10/batchnorm/mulMulAfeed_forward_sub_net_1/batch_normalization_10/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul�
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul_1Mul/feed_forward_sub_net_1/dense_8/MatMul:product:0?feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul_1�
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul_2MulIfeed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOp:value:0?feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/mul_2�
;feed_forward_sub_net_1/batch_normalization_10/batchnorm/subSubKfeed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_1/batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_1/batch_normalization_10/batchnorm/sub�
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/add_1AddV2Afeed_forward_sub_net_1/batch_normalization_10/batchnorm/mul_1:z:0?feed_forward_sub_net_1/batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_1/batch_normalization_10/batchnorm/add_1�
feed_forward_sub_net_1/Relu_3ReluAfeed_forward_sub_net_1/batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_1/Relu_3�
4feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOpReadVariableOp=feed_forward_sub_net_1_dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype026
4feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOp�
%feed_forward_sub_net_1/dense_9/MatMulMatMul+feed_forward_sub_net_1/Relu_3:activations:0<feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2'
%feed_forward_sub_net_1/dense_9/MatMul�
5feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp>feed_forward_sub_net_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype027
5feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOp�
&feed_forward_sub_net_1/dense_9/BiasAddBiasAdd/feed_forward_sub_net_1/dense_9/MatMul:product:0=feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2(
&feed_forward_sub_net_1/dense_9/BiasAdd�
IdentityIdentity/feed_forward_sub_net_1/dense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOpB^feed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOpD^feed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOpD^feed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOpD^feed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOpA^feed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOpA^feed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOpA^feed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOpA^feed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOpC^feed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOp5^feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOp5^feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOp5^feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOp5^feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOp6^feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOp5^feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2�
Afeed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOpAfeed_forward_sub_net_1/batch_normalization_10/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOpCfeed_forward_sub_net_1/batch_normalization_10/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOpCfeed_forward_sub_net_1/batch_normalization_10/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOpCfeed_forward_sub_net_1/batch_normalization_10/Cast_3/ReadVariableOp2�
@feed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOp@feed_forward_sub_net_1/batch_normalization_6/Cast/ReadVariableOp2�
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_6/Cast_1/ReadVariableOp2�
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_6/Cast_2/ReadVariableOp2�
Bfeed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_6/Cast_3/ReadVariableOp2�
@feed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOp@feed_forward_sub_net_1/batch_normalization_7/Cast/ReadVariableOp2�
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_7/Cast_1/ReadVariableOp2�
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_7/Cast_2/ReadVariableOp2�
Bfeed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_7/Cast_3/ReadVariableOp2�
@feed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOp@feed_forward_sub_net_1/batch_normalization_8/Cast/ReadVariableOp2�
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_8/Cast_1/ReadVariableOp2�
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_8/Cast_2/ReadVariableOp2�
Bfeed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_8/Cast_3/ReadVariableOp2�
@feed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOp@feed_forward_sub_net_1/batch_normalization_9/Cast/ReadVariableOp2�
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_9/Cast_1/ReadVariableOp2�
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_9/Cast_2/ReadVariableOp2�
Bfeed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOpBfeed_forward_sub_net_1/batch_normalization_9/Cast_3/ReadVariableOp2l
4feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOp4feed_forward_sub_net_1/dense_5/MatMul/ReadVariableOp2l
4feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOp4feed_forward_sub_net_1/dense_6/MatMul/ReadVariableOp2l
4feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOp4feed_forward_sub_net_1/dense_7/MatMul/ReadVariableOp2l
4feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOp4feed_forward_sub_net_1/dense_8/MatMul/ReadVariableOp2n
5feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOp5feed_forward_sub_net_1/dense_9/BiasAdd/ReadVariableOp2l
4feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOp4feed_forward_sub_net_1/dense_9/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
��
�
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_307353
input_1@
2batch_normalization_6_cast_readvariableop_resource:
B
4batch_normalization_6_cast_1_readvariableop_resource:
B
4batch_normalization_6_cast_2_readvariableop_resource:
B
4batch_normalization_6_cast_3_readvariableop_resource:
8
&dense_5_matmul_readvariableop_resource:
@
2batch_normalization_7_cast_readvariableop_resource:B
4batch_normalization_7_cast_1_readvariableop_resource:B
4batch_normalization_7_cast_2_readvariableop_resource:B
4batch_normalization_7_cast_3_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:@
2batch_normalization_8_cast_readvariableop_resource:B
4batch_normalization_8_cast_1_readvariableop_resource:B
4batch_normalization_8_cast_2_readvariableop_resource:B
4batch_normalization_8_cast_3_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:@
2batch_normalization_9_cast_readvariableop_resource:B
4batch_normalization_9_cast_1_readvariableop_resource:B
4batch_normalization_9_cast_2_readvariableop_resource:B
4batch_normalization_9_cast_3_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:A
3batch_normalization_10_cast_readvariableop_resource:C
5batch_normalization_10_cast_1_readvariableop_resource:C
5batch_normalization_10_cast_2_readvariableop_resource:C
5batch_normalization_10_cast_3_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:
5
'dense_9_biasadd_readvariableop_resource:

identity��*batch_normalization_10/Cast/ReadVariableOp�,batch_normalization_10/Cast_1/ReadVariableOp�,batch_normalization_10/Cast_2/ReadVariableOp�,batch_normalization_10/Cast_3/ReadVariableOp�)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�+batch_normalization_6/Cast_2/ReadVariableOp�+batch_normalization_6/Cast_3/ReadVariableOp�)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�+batch_normalization_7/Cast_2/ReadVariableOp�+batch_normalization_7/Cast_3/ReadVariableOp�)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�+batch_normalization_8/Cast_2/ReadVariableOp�+batch_normalization_8/Cast_3/ReadVariableOp�)batch_normalization_9/Cast/ReadVariableOp�+batch_normalization_9/Cast_1/ReadVariableOp�+batch_normalization_9/Cast_2/ReadVariableOp�+batch_normalization_9/Cast_3/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/MatMul/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:
*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
+batch_normalization_6/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02-
+batch_normalization_6/Cast_2/ReadVariableOp�
+batch_normalization_6/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02-
+batch_normalization_6/Cast_3/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV23batch_normalization_6/Cast_1/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Mulinput_1'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul1batch_normalization_6/Cast/ReadVariableOp:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub3batch_normalization_6/Cast_2/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2'
%batch_normalization_6/batchnorm/add_1�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/MatMul�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
+batch_normalization_7/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_2/ReadVariableOp�
+batch_normalization_7/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_3/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV23batch_normalization_7/Cast_1/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_5/MatMul:product:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul1batch_normalization_7/Cast/ReadVariableOp:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub3batch_normalization_7/Cast_2/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_7/batchnorm/add_1q
ReluRelu)batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp�
dense_6/MatMulMatMulRelu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6/MatMul�
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp�
+batch_normalization_8/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_2/ReadVariableOp�
+batch_normalization_8/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_3/ReadVariableOp�
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_8/batchnorm/add/y�
#batch_normalization_8/batchnorm/addAddV23batch_normalization_8/Cast_1/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/add�
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/Rsqrt�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/mul�
%batch_normalization_8/batchnorm/mul_1Muldense_6/MatMul:product:0'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_8/batchnorm/mul_1�
%batch_normalization_8/batchnorm/mul_2Mul1batch_normalization_8/Cast/ReadVariableOp:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/mul_2�
#batch_normalization_8/batchnorm/subSub3batch_normalization_8/Cast_2/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/sub�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_8/batchnorm/add_1u
Relu_1Relu)batch_normalization_8/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp�
dense_7/MatMulMatMulRelu_1:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_7/MatMul�
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp�
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp�
+batch_normalization_9/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_9_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_2/ReadVariableOp�
+batch_normalization_9/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_9_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_3/ReadVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV23batch_normalization_9/Cast_1/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/Rsqrt�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Muldense_7/MatMul:product:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_9/batchnorm/mul_1�
%batch_normalization_9/batchnorm/mul_2Mul1batch_normalization_9/Cast/ReadVariableOp:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/mul_2�
#batch_normalization_9/batchnorm/subSub3batch_normalization_9/Cast_2/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_9/batchnorm/add_1u
Relu_2Relu)batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulRelu_2:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_8/MatMul�
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp�
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp�
,batch_normalization_10/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_10_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_2/ReadVariableOp�
,batch_normalization_10/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_10_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_3/ReadVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV24batch_normalization_10/Cast_1/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/Rsqrt�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_8/MatMul:product:0(batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_10/batchnorm/mul_1�
&batch_normalization_10/batchnorm/mul_2Mul2batch_normalization_10/Cast/ReadVariableOp:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/mul_2�
$batch_normalization_10/batchnorm/subSub4batch_normalization_10/Cast_2/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_10/batchnorm/add_1v
Relu_3Relu*batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulRelu_3:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_9/BiasAdds
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�	
NoOpNoOp+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp-^batch_normalization_10/Cast_2/ReadVariableOp-^batch_normalization_10/Cast_3/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp,^batch_normalization_6/Cast_2/ReadVariableOp,^batch_normalization_6/Cast_3/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp,^batch_normalization_7/Cast_2/ReadVariableOp,^batch_normalization_7/Cast_3/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp,^batch_normalization_8/Cast_2/ReadVariableOp,^batch_normalization_8/Cast_3/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp,^batch_normalization_9/Cast_2/ReadVariableOp,^batch_normalization_9/Cast_3/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
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
:���������

!
_user_specified_name	input_1
�
�
7__inference_feed_forward_sub_net_1_layer_call_fn_307596
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
identity��StatefulPartitionedCall�
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
:���������
*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_3064182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
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
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
�
�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307787

inputs*
cast_readvariableop_resource:
,
cast_1_readvariableop_resource:
,
cast_2_readvariableop_resource:
,
cast_3_readvariableop_resource:

identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast_2/ReadVariableOp�
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
valueB 2�����ư>2
batchnorm/add/y�
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
:���������
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_305495

inputs*
cast_readvariableop_resource:
,
cast_1_readvariableop_resource:
,
cast_2_readvariableop_resource:
,
cast_3_readvariableop_resource:

identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:
*
dtype02
Cast_2/ReadVariableOp�
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
valueB 2�����ư>2
batchnorm/add/y�
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
:���������
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_308033

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_306955
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
identity��StatefulPartitionedCall�
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
:���������
*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_3054712
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
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
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
�
�
C__inference_dense_8_layer_call_and_return_conditional_losses_308226

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_307539
input_1K
=batch_normalization_6_assignmovingavg_readvariableop_resource:
M
?batch_normalization_6_assignmovingavg_1_readvariableop_resource:
@
2batch_normalization_6_cast_readvariableop_resource:
B
4batch_normalization_6_cast_1_readvariableop_resource:
8
&dense_5_matmul_readvariableop_resource:
K
=batch_normalization_7_assignmovingavg_readvariableop_resource:M
?batch_normalization_7_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_7_cast_readvariableop_resource:B
4batch_normalization_7_cast_1_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:K
=batch_normalization_8_assignmovingavg_readvariableop_resource:M
?batch_normalization_8_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_8_cast_readvariableop_resource:B
4batch_normalization_8_cast_1_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:K
=batch_normalization_9_assignmovingavg_readvariableop_resource:M
?batch_normalization_9_assignmovingavg_1_readvariableop_resource:@
2batch_normalization_9_cast_readvariableop_resource:B
4batch_normalization_9_cast_1_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:L
>batch_normalization_10_assignmovingavg_readvariableop_resource:N
@batch_normalization_10_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_10_cast_readvariableop_resource:C
5batch_normalization_10_cast_1_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:
5
'dense_9_biasadd_readvariableop_resource:

identity��&batch_normalization_10/AssignMovingAvg�5batch_normalization_10/AssignMovingAvg/ReadVariableOp�(batch_normalization_10/AssignMovingAvg_1�7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_10/Cast/ReadVariableOp�,batch_normalization_10/Cast_1/ReadVariableOp�%batch_normalization_6/AssignMovingAvg�4batch_normalization_6/AssignMovingAvg/ReadVariableOp�'batch_normalization_6/AssignMovingAvg_1�6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�%batch_normalization_7/AssignMovingAvg�4batch_normalization_7/AssignMovingAvg/ReadVariableOp�'batch_normalization_7/AssignMovingAvg_1�6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�%batch_normalization_8/AssignMovingAvg�4batch_normalization_8/AssignMovingAvg/ReadVariableOp�'batch_normalization_8/AssignMovingAvg_1�6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�%batch_normalization_9/AssignMovingAvg�4batch_normalization_9/AssignMovingAvg/ReadVariableOp�'batch_normalization_9/AssignMovingAvg_1�6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_9/Cast/ReadVariableOp�+batch_normalization_9/Cast_1/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/MatMul/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_6/moments/mean/reduction_indices�
"batch_normalization_6/moments/meanMeaninput_1=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2$
"batch_normalization_6/moments/mean�
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes

:
2,
*batch_normalization_6/moments/StopGradient�
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferenceinput_13batch_normalization_6/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������
21
/batch_normalization_6/moments/SquaredDifference�
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_6/moments/variance/reduction_indices�
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2(
&batch_normalization_6/moments/variance�
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2'
%batch_normalization_6/moments/Squeeze�
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1�
+batch_normalization_6/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_6/AssignMovingAvg/decay�
*batch_normalization_6/AssignMovingAvg/CastCast4batch_normalization_6/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_6/AssignMovingAvg/Cast�
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOp�
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*
_output_shapes
:
2+
)batch_normalization_6/AssignMovingAvg/sub�
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:0.batch_normalization_6/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2+
)batch_normalization_6/AssignMovingAvg/mul�
%batch_normalization_6/AssignMovingAvgAssignSubVariableOp=batch_normalization_6_assignmovingavg_readvariableop_resource-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_6/AssignMovingAvg�
-batch_normalization_6/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_6/AssignMovingAvg_1/decay�
,batch_normalization_6/AssignMovingAvg_1/CastCast6batch_normalization_6/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_6/AssignMovingAvg_1/Cast�
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2-
+batch_normalization_6/AssignMovingAvg_1/sub�
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:00batch_normalization_6/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2-
+batch_normalization_6/AssignMovingAvg_1/mul�
'batch_normalization_6/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_6_assignmovingavg_1_readvariableop_resource/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_6/AssignMovingAvg_1�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:
*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Mulinput_1'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub1batch_normalization_6/Cast/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2'
%batch_normalization_6/batchnorm/add_1�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/MatMul�
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_7/moments/mean/reduction_indices�
"batch_normalization_7/moments/meanMeandense_5/MatMul:product:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_7/moments/mean�
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_7/moments/StopGradient�
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferencedense_5/MatMul:product:03batch_normalization_7/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������21
/batch_normalization_7/moments/SquaredDifference�
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_7/moments/variance/reduction_indices�
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_7/moments/variance�
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_7/moments/Squeeze�
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1�
+batch_normalization_7/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_7/AssignMovingAvg/decay�
*batch_normalization_7/AssignMovingAvg/CastCast4batch_normalization_7/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_7/AssignMovingAvg/Cast�
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOp�
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/sub�
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:0.batch_normalization_7/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/mul�
%batch_normalization_7/AssignMovingAvgAssignSubVariableOp=batch_normalization_7_assignmovingavg_readvariableop_resource-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_7/AssignMovingAvg�
-batch_normalization_7/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_7/AssignMovingAvg_1/decay�
,batch_normalization_7/AssignMovingAvg_1/CastCast6batch_normalization_7/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_7/AssignMovingAvg_1/Cast�
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/sub�
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:00batch_normalization_7/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/mul�
'batch_normalization_7/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_7_assignmovingavg_1_readvariableop_resource/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_7/AssignMovingAvg_1�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_5/MatMul:product:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub1batch_normalization_7/Cast/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_7/batchnorm/add_1q
ReluRelu)batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp�
dense_6/MatMulMatMulRelu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6/MatMul�
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_8/moments/mean/reduction_indices�
"batch_normalization_8/moments/meanMeandense_6/MatMul:product:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_8/moments/mean�
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_8/moments/StopGradient�
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferencedense_6/MatMul:product:03batch_normalization_8/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������21
/batch_normalization_8/moments/SquaredDifference�
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_8/moments/variance/reduction_indices�
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_8/moments/variance�
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_8/moments/Squeeze�
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1�
+batch_normalization_8/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_8/AssignMovingAvg/decay�
*batch_normalization_8/AssignMovingAvg/CastCast4batch_normalization_8/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_8/AssignMovingAvg/Cast�
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOp�
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_8/AssignMovingAvg/sub�
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:0.batch_normalization_8/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_8/AssignMovingAvg/mul�
%batch_normalization_8/AssignMovingAvgAssignSubVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_8/AssignMovingAvg�
-batch_normalization_8/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_8/AssignMovingAvg_1/decay�
,batch_normalization_8/AssignMovingAvg_1/CastCast6batch_normalization_8/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_8/AssignMovingAvg_1/Cast�
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_8/AssignMovingAvg_1/sub�
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:00batch_normalization_8/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_8/AssignMovingAvg_1/mul�
'batch_normalization_8/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_8/AssignMovingAvg_1�
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp�
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_8/batchnorm/add/y�
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/add�
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/Rsqrt�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/mul�
%batch_normalization_8/batchnorm/mul_1Muldense_6/MatMul:product:0'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_8/batchnorm/mul_1�
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/mul_2�
#batch_normalization_8/batchnorm/subSub1batch_normalization_8/Cast/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/sub�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_8/batchnorm/add_1u
Relu_1Relu)batch_normalization_8/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp�
dense_7/MatMulMatMulRelu_1:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_7/MatMul�
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_9/moments/mean/reduction_indices�
"batch_normalization_9/moments/meanMeandense_7/MatMul:product:0=batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2$
"batch_normalization_9/moments/mean�
*batch_normalization_9/moments/StopGradientStopGradient+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes

:2,
*batch_normalization_9/moments/StopGradient�
/batch_normalization_9/moments/SquaredDifferenceSquaredDifferencedense_7/MatMul:product:03batch_normalization_9/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������21
/batch_normalization_9/moments/SquaredDifference�
8batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_9/moments/variance/reduction_indices�
&batch_normalization_9/moments/varianceMean3batch_normalization_9/moments/SquaredDifference:z:0Abatch_normalization_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2(
&batch_normalization_9/moments/variance�
%batch_normalization_9/moments/SqueezeSqueeze+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_9/moments/Squeeze�
'batch_normalization_9/moments/Squeeze_1Squeeze/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_9/moments/Squeeze_1�
+batch_normalization_9/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_9/AssignMovingAvg/decay�
*batch_normalization_9/AssignMovingAvg/CastCast4batch_normalization_9/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*batch_normalization_9/AssignMovingAvg/Cast�
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_9_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOp�
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0*
T0*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/sub�
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:0.batch_normalization_9/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2+
)batch_normalization_9/AssignMovingAvg/mul�
%batch_normalization_9/AssignMovingAvgAssignSubVariableOp=batch_normalization_9_assignmovingavg_readvariableop_resource-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_9/AssignMovingAvg�
-batch_normalization_9/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_9/AssignMovingAvg_1/decay�
,batch_normalization_9/AssignMovingAvg_1/CastCast6batch_normalization_9/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,batch_normalization_9/AssignMovingAvg_1/Cast�
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_9_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/sub�
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:00batch_normalization_9/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2-
+batch_normalization_9/AssignMovingAvg_1/mul�
'batch_normalization_9/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_9_assignmovingavg_1_readvariableop_resource/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_9/AssignMovingAvg_1�
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp�
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV20batch_normalization_9/moments/Squeeze_1:output:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/Rsqrt�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Muldense_7/MatMul:product:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_9/batchnorm/mul_1�
%batch_normalization_9/batchnorm/mul_2Mul.batch_normalization_9/moments/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/mul_2�
#batch_normalization_9/batchnorm/subSub1batch_normalization_9/Cast/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_9/batchnorm/add_1u
Relu_2Relu)batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulRelu_2:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_8/MatMul�
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_10/moments/mean/reduction_indices�
#batch_normalization_10/moments/meanMeandense_8/MatMul:product:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_10/moments/mean�
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_10/moments/StopGradient�
0batch_normalization_10/moments/SquaredDifferenceSquaredDifferencedense_8/MatMul:product:04batch_normalization_10/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_10/moments/SquaredDifference�
9batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_10/moments/variance/reduction_indices�
'batch_normalization_10/moments/varianceMean4batch_normalization_10/moments/SquaredDifference:z:0Bbatch_normalization_10/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_10/moments/variance�
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_10/moments/Squeeze�
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_10/moments/Squeeze_1�
,batch_normalization_10/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_10/AssignMovingAvg/decay�
+batch_normalization_10/AssignMovingAvg/CastCast5batch_normalization_10/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_10/AssignMovingAvg/Cast�
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_10_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOp�
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/sub�
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:0/batch_normalization_10/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_10/AssignMovingAvg/mul�
&batch_normalization_10/AssignMovingAvgAssignSubVariableOp>batch_normalization_10_assignmovingavg_readvariableop_resource.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_10/AssignMovingAvg�
.batch_normalization_10/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_10/AssignMovingAvg_1/decay�
-batch_normalization_10/AssignMovingAvg_1/CastCast7batch_normalization_10/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_10/AssignMovingAvg_1/Cast�
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_10_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/sub�
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:01batch_normalization_10/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_10/AssignMovingAvg_1/mul�
(batch_normalization_10/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_10_assignmovingavg_1_readvariableop_resource0batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_10/AssignMovingAvg_1�
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp�
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV21batch_normalization_10/moments/Squeeze_1:output:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/Rsqrt�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_8/MatMul:product:0(batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_10/batchnorm/mul_1�
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/mul_2�
$batch_normalization_10/batchnorm/subSub2batch_normalization_10/Cast/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_10/batchnorm/add_1v
Relu_3Relu*batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulRelu_3:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_9/BiasAdds
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp'^batch_normalization_10/AssignMovingAvg6^batch_normalization_10/AssignMovingAvg/ReadVariableOp)^batch_normalization_10/AssignMovingAvg_18^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp&^batch_normalization_6/AssignMovingAvg5^batch_normalization_6/AssignMovingAvg/ReadVariableOp(^batch_normalization_6/AssignMovingAvg_17^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp&^batch_normalization_7/AssignMovingAvg5^batch_normalization_7/AssignMovingAvg/ReadVariableOp(^batch_normalization_7/AssignMovingAvg_17^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp&^batch_normalization_8/AssignMovingAvg5^batch_normalization_8/AssignMovingAvg/ReadVariableOp(^batch_normalization_8/AssignMovingAvg_17^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp&^batch_normalization_9/AssignMovingAvg5^batch_normalization_9/AssignMovingAvg/ReadVariableOp(^batch_normalization_9/AssignMovingAvg_17^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
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
:���������

!
_user_specified_name	input_1
�

�
C__inference_dense_9_layer_call_and_return_conditional_losses_306411

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
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
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_307061
x@
2batch_normalization_6_cast_readvariableop_resource:
B
4batch_normalization_6_cast_1_readvariableop_resource:
B
4batch_normalization_6_cast_2_readvariableop_resource:
B
4batch_normalization_6_cast_3_readvariableop_resource:
8
&dense_5_matmul_readvariableop_resource:
@
2batch_normalization_7_cast_readvariableop_resource:B
4batch_normalization_7_cast_1_readvariableop_resource:B
4batch_normalization_7_cast_2_readvariableop_resource:B
4batch_normalization_7_cast_3_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:@
2batch_normalization_8_cast_readvariableop_resource:B
4batch_normalization_8_cast_1_readvariableop_resource:B
4batch_normalization_8_cast_2_readvariableop_resource:B
4batch_normalization_8_cast_3_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:@
2batch_normalization_9_cast_readvariableop_resource:B
4batch_normalization_9_cast_1_readvariableop_resource:B
4batch_normalization_9_cast_2_readvariableop_resource:B
4batch_normalization_9_cast_3_readvariableop_resource:8
&dense_8_matmul_readvariableop_resource:A
3batch_normalization_10_cast_readvariableop_resource:C
5batch_normalization_10_cast_1_readvariableop_resource:C
5batch_normalization_10_cast_2_readvariableop_resource:C
5batch_normalization_10_cast_3_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:
5
'dense_9_biasadd_readvariableop_resource:

identity��*batch_normalization_10/Cast/ReadVariableOp�,batch_normalization_10/Cast_1/ReadVariableOp�,batch_normalization_10/Cast_2/ReadVariableOp�,batch_normalization_10/Cast_3/ReadVariableOp�)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�+batch_normalization_6/Cast_2/ReadVariableOp�+batch_normalization_6/Cast_3/ReadVariableOp�)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�+batch_normalization_7/Cast_2/ReadVariableOp�+batch_normalization_7/Cast_3/ReadVariableOp�)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�+batch_normalization_8/Cast_2/ReadVariableOp�+batch_normalization_8/Cast_3/ReadVariableOp�)batch_normalization_9/Cast/ReadVariableOp�+batch_normalization_9/Cast_1/ReadVariableOp�+batch_normalization_9/Cast_2/ReadVariableOp�+batch_normalization_9/Cast_3/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/MatMul/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes
:
*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
+batch_normalization_6/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02-
+batch_normalization_6/Cast_2/ReadVariableOp�
+batch_normalization_6/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02-
+batch_normalization_6/Cast_3/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV23batch_normalization_6/Cast_1/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Mulx'batch_normalization_6/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul1batch_normalization_6/Cast/ReadVariableOp:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:
2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub3batch_normalization_6/Cast_2/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2'
%batch_normalization_6/batchnorm/add_1�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMul)batch_normalization_6/batchnorm/add_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/MatMul�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
+batch_normalization_7/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_2/ReadVariableOp�
+batch_normalization_7/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_7/Cast_3/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV23batch_normalization_7/Cast_1/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_5/MatMul:product:0'batch_normalization_7/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul1batch_normalization_7/Cast/ReadVariableOp:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub3batch_normalization_7/Cast_2/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_7/batchnorm/add_1q
ReluRelu)batch_normalization_7/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp�
dense_6/MatMulMatMulRelu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_6/MatMul�
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp�
+batch_normalization_8/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_2/ReadVariableOp�
+batch_normalization_8/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_8/Cast_3/ReadVariableOp�
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_8/batchnorm/add/y�
#batch_normalization_8/batchnorm/addAddV23batch_normalization_8/Cast_1/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/add�
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/Rsqrt�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/mul�
%batch_normalization_8/batchnorm/mul_1Muldense_6/MatMul:product:0'batch_normalization_8/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_8/batchnorm/mul_1�
%batch_normalization_8/batchnorm/mul_2Mul1batch_normalization_8/Cast/ReadVariableOp:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_8/batchnorm/mul_2�
#batch_normalization_8/batchnorm/subSub3batch_normalization_8/Cast_2/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_8/batchnorm/sub�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_8/batchnorm/add_1u
Relu_1Relu)batch_normalization_8/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp�
dense_7/MatMulMatMulRelu_1:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_7/MatMul�
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp�
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp�
+batch_normalization_9/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_9_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_2/ReadVariableOp�
+batch_normalization_9/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_9_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batch_normalization_9/Cast_3/ReadVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV23batch_normalization_9/Cast_1/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/Rsqrt�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Muldense_7/MatMul:product:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_9/batchnorm/mul_1�
%batch_normalization_9/batchnorm/mul_2Mul1batch_normalization_9/Cast/ReadVariableOp:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_9/batchnorm/mul_2�
#batch_normalization_9/batchnorm/subSub3batch_normalization_9/Cast_2/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2'
%batch_normalization_9/batchnorm/add_1u
Relu_2Relu)batch_normalization_9/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulRelu_2:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_8/MatMul�
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp�
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp�
,batch_normalization_10/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_10_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_2/ReadVariableOp�
,batch_normalization_10/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_10_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_10/Cast_3/ReadVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV24batch_normalization_10/Cast_1/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/Rsqrt�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_8/MatMul:product:0(batch_normalization_10/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_10/batchnorm/mul_1�
&batch_normalization_10/batchnorm/mul_2Mul2batch_normalization_10/Cast/ReadVariableOp:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_10/batchnorm/mul_2�
$batch_normalization_10/batchnorm/subSub4batch_normalization_10/Cast_2/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_10/batchnorm/add_1v
Relu_3Relu*batch_normalization_10/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulRelu_3:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_9/BiasAdds
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�	
NoOpNoOp+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp-^batch_normalization_10/Cast_2/ReadVariableOp-^batch_normalization_10/Cast_3/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp,^batch_normalization_6/Cast_2/ReadVariableOp,^batch_normalization_6/Cast_3/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp,^batch_normalization_7/Cast_2/ReadVariableOp,^batch_normalization_7/Cast_3/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp,^batch_normalization_8/Cast_2/ReadVariableOp,^batch_normalization_8/Cast_3/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp,^batch_normalization_9/Cast_2/ReadVariableOp,^batch_normalization_9/Cast_3/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
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
:���������


_user_specified_namex
�
|
(__inference_dense_5_layer_call_fn_308191

inputs
unknown:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3063242
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������
: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_307951

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�C
�
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_306644
x*
batch_normalization_6_306577:
*
batch_normalization_6_306579:
*
batch_normalization_6_306581:
*
batch_normalization_6_306583:
 
dense_5_306586:
*
batch_normalization_7_306589:*
batch_normalization_7_306591:*
batch_normalization_7_306593:*
batch_normalization_7_306595: 
dense_6_306599:*
batch_normalization_8_306602:*
batch_normalization_8_306604:*
batch_normalization_8_306606:*
batch_normalization_8_306608: 
dense_7_306612:*
batch_normalization_9_306615:*
batch_normalization_9_306617:*
batch_normalization_9_306619:*
batch_normalization_9_306621: 
dense_8_306625:+
batch_normalization_10_306628:+
batch_normalization_10_306630:+
batch_normalization_10_306632:+
batch_normalization_10_306634: 
dense_9_306638:

dense_9_306640:

identity��.batch_normalization_10/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_6_306577batch_normalization_6_306579batch_normalization_6_306581batch_normalization_6_306583*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_3055572/
-batch_normalization_6/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0dense_5_306586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_3063242!
dense_5/StatefulPartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0batch_normalization_7_306589batch_normalization_7_306591batch_normalization_7_306593batch_normalization_7_306595*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_3057232/
-batch_normalization_7/StatefulPartitionedCall~
ReluRelu6batch_normalization_7/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu�
dense_6/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_6_306599*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_3063452!
dense_6/StatefulPartitionedCall�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0batch_normalization_8_306602batch_normalization_8_306604batch_normalization_8_306606batch_normalization_8_306608*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_3058892/
-batch_normalization_8/StatefulPartitionedCall�
Relu_1Relu6batch_normalization_8/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_7/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_7_306612*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_3063662!
dense_7/StatefulPartitionedCall�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0batch_normalization_9_306615batch_normalization_9_306617batch_normalization_9_306619batch_normalization_9_306621*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3060552/
-batch_normalization_9/StatefulPartitionedCall�
Relu_2Relu6batch_normalization_9/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_8/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_8_306625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_3063872!
dense_8/StatefulPartitionedCall�
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0batch_normalization_10_306628batch_normalization_10_306630batch_normalization_10_306632batch_normalization_10_306634*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_30622120
.batch_normalization_10/StatefulPartitionedCall�
Relu_3Relu7batch_normalization_10/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_9/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_9_306638dense_9_306640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3064112!
dense_9/StatefulPartitionedCall�
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
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
:���������


_user_specified_namex
�z
�
"__inference__traced_restore_308441
file_prefixQ
Cassignvariableop_feed_forward_sub_net_1_batch_normalization_6_gamma:
R
Dassignvariableop_1_feed_forward_sub_net_1_batch_normalization_6_beta:
S
Eassignvariableop_2_feed_forward_sub_net_1_batch_normalization_7_gamma:R
Dassignvariableop_3_feed_forward_sub_net_1_batch_normalization_7_beta:S
Eassignvariableop_4_feed_forward_sub_net_1_batch_normalization_8_gamma:R
Dassignvariableop_5_feed_forward_sub_net_1_batch_normalization_8_beta:S
Eassignvariableop_6_feed_forward_sub_net_1_batch_normalization_9_gamma:R
Dassignvariableop_7_feed_forward_sub_net_1_batch_normalization_9_beta:T
Fassignvariableop_8_feed_forward_sub_net_1_batch_normalization_10_gamma:S
Eassignvariableop_9_feed_forward_sub_net_1_batch_normalization_10_beta:Z
Lassignvariableop_10_feed_forward_sub_net_1_batch_normalization_6_moving_mean:
^
Passignvariableop_11_feed_forward_sub_net_1_batch_normalization_6_moving_variance:
Z
Lassignvariableop_12_feed_forward_sub_net_1_batch_normalization_7_moving_mean:^
Passignvariableop_13_feed_forward_sub_net_1_batch_normalization_7_moving_variance:Z
Lassignvariableop_14_feed_forward_sub_net_1_batch_normalization_8_moving_mean:^
Passignvariableop_15_feed_forward_sub_net_1_batch_normalization_8_moving_variance:Z
Lassignvariableop_16_feed_forward_sub_net_1_batch_normalization_9_moving_mean:^
Passignvariableop_17_feed_forward_sub_net_1_batch_normalization_9_moving_variance:[
Massignvariableop_18_feed_forward_sub_net_1_batch_normalization_10_moving_mean:_
Qassignvariableop_19_feed_forward_sub_net_1_batch_normalization_10_moving_variance:K
9assignvariableop_20_feed_forward_sub_net_1_dense_5_kernel:
K
9assignvariableop_21_feed_forward_sub_net_1_dense_6_kernel:K
9assignvariableop_22_feed_forward_sub_net_1_dense_7_kernel:K
9assignvariableop_23_feed_forward_sub_net_1_dense_8_kernel:K
9assignvariableop_24_feed_forward_sub_net_1_dense_9_kernel:
E
7assignvariableop_25_feed_forward_sub_net_1_dense_9_bias:

identity_27��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpCassignvariableop_feed_forward_sub_net_1_batch_normalization_6_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpDassignvariableop_1_feed_forward_sub_net_1_batch_normalization_6_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpEassignvariableop_2_feed_forward_sub_net_1_batch_normalization_7_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpDassignvariableop_3_feed_forward_sub_net_1_batch_normalization_7_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpEassignvariableop_4_feed_forward_sub_net_1_batch_normalization_8_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpDassignvariableop_5_feed_forward_sub_net_1_batch_normalization_8_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpEassignvariableop_6_feed_forward_sub_net_1_batch_normalization_9_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpDassignvariableop_7_feed_forward_sub_net_1_batch_normalization_9_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpFassignvariableop_8_feed_forward_sub_net_1_batch_normalization_10_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpEassignvariableop_9_feed_forward_sub_net_1_batch_normalization_10_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpLassignvariableop_10_feed_forward_sub_net_1_batch_normalization_6_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpPassignvariableop_11_feed_forward_sub_net_1_batch_normalization_6_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpLassignvariableop_12_feed_forward_sub_net_1_batch_normalization_7_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpPassignvariableop_13_feed_forward_sub_net_1_batch_normalization_7_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpLassignvariableop_14_feed_forward_sub_net_1_batch_normalization_8_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpPassignvariableop_15_feed_forward_sub_net_1_batch_normalization_8_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpLassignvariableop_16_feed_forward_sub_net_1_batch_normalization_9_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpPassignvariableop_17_feed_forward_sub_net_1_batch_normalization_9_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpMassignvariableop_18_feed_forward_sub_net_1_batch_normalization_10_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpQassignvariableop_19_feed_forward_sub_net_1_batch_normalization_10_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp9assignvariableop_20_feed_forward_sub_net_1_dense_5_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp9assignvariableop_21_feed_forward_sub_net_1_dense_6_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp9assignvariableop_22_feed_forward_sub_net_1_dense_7_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp9assignvariableop_23_feed_forward_sub_net_1_dense_8_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp9assignvariableop_24_feed_forward_sub_net_1_dense_9_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp7assignvariableop_25_feed_forward_sub_net_1_dense_9_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_259
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_26f
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_27�
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
�
�
C__inference_dense_7_layer_call_and_return_conditional_losses_308212

inputs0
matmul_readvariableop_resource:
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_308115

inputs*
cast_readvariableop_resource:,
cast_1_readvariableop_resource:,
cast_2_readvariableop_resource:,
cast_3_readvariableop_resource:
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
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
:���������2
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
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_9_layer_call_fn_308082

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3059932
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������
<
output_10
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
	bn_layers
dense_layers
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�_default_save_signature
+�&call_and_return_all_conditional_losses
�__call__"
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
�
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
�
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
�
-metrics

.layers
	variables
trainable_variables
regularization_losses
/non_trainable_variables
0layer_regularization_losses
1layer_metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�
2axis
	gamma
beta
moving_mean
moving_variance
3	variables
4trainable_variables
5regularization_losses
6	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
7axis
	gamma
beta
moving_mean
 moving_variance
8	variables
9trainable_variables
:regularization_losses
;	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
<axis
	gamma
beta
!moving_mean
"moving_variance
=	variables
>trainable_variables
?regularization_losses
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Aaxis
	gamma
beta
#moving_mean
$moving_variance
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Faxis
	gamma
beta
%moving_mean
&moving_variance
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
�

'kernel
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

(kernel
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

)kernel
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

*kernel
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

+kernel
,bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
@:>
22feed_forward_sub_net_1/batch_normalization_6/gamma
?:=
21feed_forward_sub_net_1/batch_normalization_6/beta
@:>22feed_forward_sub_net_1/batch_normalization_7/gamma
?:=21feed_forward_sub_net_1/batch_normalization_7/beta
@:>22feed_forward_sub_net_1/batch_normalization_8/gamma
?:=21feed_forward_sub_net_1/batch_normalization_8/beta
@:>22feed_forward_sub_net_1/batch_normalization_9/gamma
?:=21feed_forward_sub_net_1/batch_normalization_9/beta
A:?23feed_forward_sub_net_1/batch_normalization_10/gamma
@:>22feed_forward_sub_net_1/batch_normalization_10/beta
H:F
 (28feed_forward_sub_net_1/batch_normalization_6/moving_mean
L:J
 (2<feed_forward_sub_net_1/batch_normalization_6/moving_variance
H:F (28feed_forward_sub_net_1/batch_normalization_7/moving_mean
L:J (2<feed_forward_sub_net_1/batch_normalization_7/moving_variance
H:F (28feed_forward_sub_net_1/batch_normalization_8/moving_mean
L:J (2<feed_forward_sub_net_1/batch_normalization_8/moving_variance
H:F (28feed_forward_sub_net_1/batch_normalization_9/moving_mean
L:J (2<feed_forward_sub_net_1/batch_normalization_9/moving_variance
I:G (29feed_forward_sub_net_1/batch_normalization_10/moving_mean
M:K (2=feed_forward_sub_net_1/batch_normalization_10/moving_variance
7:5
2%feed_forward_sub_net_1/dense_5/kernel
7:52%feed_forward_sub_net_1/dense_6/kernel
7:52%feed_forward_sub_net_1/dense_7/kernel
7:52%feed_forward_sub_net_1/dense_8/kernel
7:5
2%feed_forward_sub_net_1/dense_9/kernel
1:/
2#feed_forward_sub_net_1/dense_9/bias
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
�
`metrics

alayers
3	variables
4trainable_variables
5regularization_losses
bnon_trainable_variables
clayer_regularization_losses
dlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
emetrics

flayers
8	variables
9trainable_variables
:regularization_losses
gnon_trainable_variables
hlayer_regularization_losses
ilayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
jmetrics

klayers
=	variables
>trainable_variables
?regularization_losses
lnon_trainable_variables
mlayer_regularization_losses
nlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
ometrics

players
B	variables
Ctrainable_variables
Dregularization_losses
qnon_trainable_variables
rlayer_regularization_losses
slayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
tmetrics

ulayers
G	variables
Htrainable_variables
Iregularization_losses
vnon_trainable_variables
wlayer_regularization_losses
xlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
ymetrics

zlayers
L	variables
Mtrainable_variables
Nregularization_losses
{non_trainable_variables
|layer_regularization_losses
}layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
'
(0"
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
~metrics

layers
P	variables
Qtrainable_variables
Rregularization_losses
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
'
)0"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layers
T	variables
Utrainable_variables
Vregularization_losses
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�layers
X	variables
Ytrainable_variables
Zregularization_losses
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
�metrics
�layers
\	variables
]trainable_variables
^regularization_losses
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�B�
!__inference__wrapped_model_305471input_1"�
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
�2�
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_307061
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_307247
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_307353
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_307539�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_feed_forward_sub_net_1_layer_call_fn_307596
7__inference_feed_forward_sub_net_1_layer_call_fn_307653
7__inference_feed_forward_sub_net_1_layer_call_fn_307710
7__inference_feed_forward_sub_net_1_layer_call_fn_307767�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
$__inference_signature_wrapper_306955input_1"�
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
�2�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307787
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307823�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_6_layer_call_fn_307836
6__inference_batch_normalization_6_layer_call_fn_307849�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_307869
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_307905�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_7_layer_call_fn_307918
6__inference_batch_normalization_7_layer_call_fn_307931�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_307951
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_307987�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_8_layer_call_fn_308000
6__inference_batch_normalization_8_layer_call_fn_308013�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_308033
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_308069�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_9_layer_call_fn_308082
6__inference_batch_normalization_9_layer_call_fn_308095�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_308115
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_308151�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_10_layer_call_fn_308164
7__inference_batch_normalization_10_layer_call_fn_308177�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_dense_5_layer_call_and_return_conditional_losses_308184�
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
�2�
(__inference_dense_5_layer_call_fn_308191�
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
�2�
C__inference_dense_6_layer_call_and_return_conditional_losses_308198�
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
�2�
(__inference_dense_6_layer_call_fn_308205�
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
�2�
C__inference_dense_7_layer_call_and_return_conditional_losses_308212�
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
�2�
(__inference_dense_7_layer_call_fn_308219�
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
�2�
C__inference_dense_8_layer_call_and_return_conditional_losses_308226�
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
�2�
(__inference_dense_8_layer_call_fn_308233�
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
�2�
C__inference_dense_9_layer_call_and_return_conditional_losses_308243�
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
�2�
(__inference_dense_9_layer_call_fn_308252�
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
!__inference__wrapped_model_305471�' (!")#$*%&+,0�-
&�#
!�
input_1���������

� "3�0
.
output_1"�
output_1���������
�
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_308115b%&3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_308151b%&3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_10_layer_call_fn_308164U%&3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_10_layer_call_fn_308177U%&3�0
)�&
 �
inputs���������
p
� "�����������
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307787b3�0
)�&
 �
inputs���������

p 
� "%�"
�
0���������

� �
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_307823b3�0
)�&
 �
inputs���������

p
� "%�"
�
0���������

� �
6__inference_batch_normalization_6_layer_call_fn_307836U3�0
)�&
 �
inputs���������

p 
� "����������
�
6__inference_batch_normalization_6_layer_call_fn_307849U3�0
)�&
 �
inputs���������

p
� "����������
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_307869b 3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_307905b 3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
6__inference_batch_normalization_7_layer_call_fn_307918U 3�0
)�&
 �
inputs���������
p 
� "�����������
6__inference_batch_normalization_7_layer_call_fn_307931U 3�0
)�&
 �
inputs���������
p
� "�����������
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_307951b!"3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_307987b!"3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
6__inference_batch_normalization_8_layer_call_fn_308000U!"3�0
)�&
 �
inputs���������
p 
� "�����������
6__inference_batch_normalization_8_layer_call_fn_308013U!"3�0
)�&
 �
inputs���������
p
� "�����������
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_308033b#$3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_308069b#$3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
6__inference_batch_normalization_9_layer_call_fn_308082U#$3�0
)�&
 �
inputs���������
p 
� "�����������
6__inference_batch_normalization_9_layer_call_fn_308095U#$3�0
)�&
 �
inputs���������
p
� "�����������
C__inference_dense_5_layer_call_and_return_conditional_losses_308184['/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� z
(__inference_dense_5_layer_call_fn_308191N'/�,
%�"
 �
inputs���������

� "�����������
C__inference_dense_6_layer_call_and_return_conditional_losses_308198[(/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
(__inference_dense_6_layer_call_fn_308205N(/�,
%�"
 �
inputs���������
� "�����������
C__inference_dense_7_layer_call_and_return_conditional_losses_308212[)/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
(__inference_dense_7_layer_call_fn_308219N)/�,
%�"
 �
inputs���������
� "�����������
C__inference_dense_8_layer_call_and_return_conditional_losses_308226[*/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
(__inference_dense_8_layer_call_fn_308233N*/�,
%�"
 �
inputs���������
� "�����������
C__inference_dense_9_layer_call_and_return_conditional_losses_308243\+,/�,
%�"
 �
inputs���������
� "%�"
�
0���������

� {
(__inference_dense_9_layer_call_fn_308252O+,/�,
%�"
 �
inputs���������
� "����������
�
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_307061s' (!")#$*%&+,.�+
$�!
�
x���������

p 
� "%�"
�
0���������

� �
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_307247s' (!")#$*%&+,.�+
$�!
�
x���������

p
� "%�"
�
0���������

� �
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_307353y' (!")#$*%&+,4�1
*�'
!�
input_1���������

p 
� "%�"
�
0���������

� �
R__inference_feed_forward_sub_net_1_layer_call_and_return_conditional_losses_307539y' (!")#$*%&+,4�1
*�'
!�
input_1���������

p
� "%�"
�
0���������

� �
7__inference_feed_forward_sub_net_1_layer_call_fn_307596l' (!")#$*%&+,4�1
*�'
!�
input_1���������

p 
� "����������
�
7__inference_feed_forward_sub_net_1_layer_call_fn_307653f' (!")#$*%&+,.�+
$�!
�
x���������

p 
� "����������
�
7__inference_feed_forward_sub_net_1_layer_call_fn_307710f' (!")#$*%&+,.�+
$�!
�
x���������

p
� "����������
�
7__inference_feed_forward_sub_net_1_layer_call_fn_307767l' (!")#$*%&+,4�1
*�'
!�
input_1���������

p
� "����������
�
$__inference_signature_wrapper_306955�' (!")#$*%&+,;�8
� 
1�.
,
input_1!�
input_1���������
"3�0
.
output_1"�
output_1���������
