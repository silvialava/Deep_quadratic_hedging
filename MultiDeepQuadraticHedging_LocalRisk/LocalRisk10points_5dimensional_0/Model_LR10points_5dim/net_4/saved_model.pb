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
 �"serve*2.6.02unknown8֕
�
3feed_forward_sub_net_4/batch_normalization_24/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*D
shared_name53feed_forward_sub_net_4/batch_normalization_24/gamma
�
Gfeed_forward_sub_net_4/batch_normalization_24/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_4/batch_normalization_24/gamma*
_output_shapes
:
*
dtype0
�
2feed_forward_sub_net_4/batch_normalization_24/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*C
shared_name42feed_forward_sub_net_4/batch_normalization_24/beta
�
Ffeed_forward_sub_net_4/batch_normalization_24/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_4/batch_normalization_24/beta*
_output_shapes
:
*
dtype0
�
3feed_forward_sub_net_4/batch_normalization_25/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_4/batch_normalization_25/gamma
�
Gfeed_forward_sub_net_4/batch_normalization_25/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_4/batch_normalization_25/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_4/batch_normalization_25/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_4/batch_normalization_25/beta
�
Ffeed_forward_sub_net_4/batch_normalization_25/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_4/batch_normalization_25/beta*
_output_shapes
:*
dtype0
�
3feed_forward_sub_net_4/batch_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_4/batch_normalization_26/gamma
�
Gfeed_forward_sub_net_4/batch_normalization_26/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_4/batch_normalization_26/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_4/batch_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_4/batch_normalization_26/beta
�
Ffeed_forward_sub_net_4/batch_normalization_26/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_4/batch_normalization_26/beta*
_output_shapes
:*
dtype0
�
3feed_forward_sub_net_4/batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_4/batch_normalization_27/gamma
�
Gfeed_forward_sub_net_4/batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_4/batch_normalization_27/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_4/batch_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_4/batch_normalization_27/beta
�
Ffeed_forward_sub_net_4/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_4/batch_normalization_27/beta*
_output_shapes
:*
dtype0
�
3feed_forward_sub_net_4/batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_4/batch_normalization_28/gamma
�
Gfeed_forward_sub_net_4/batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_4/batch_normalization_28/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_4/batch_normalization_28/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_4/batch_normalization_28/beta
�
Ffeed_forward_sub_net_4/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_4/batch_normalization_28/beta*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_4/batch_normalization_24/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*J
shared_name;9feed_forward_sub_net_4/batch_normalization_24/moving_mean
�
Mfeed_forward_sub_net_4/batch_normalization_24/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_4/batch_normalization_24/moving_mean*
_output_shapes
:
*
dtype0
�
=feed_forward_sub_net_4/batch_normalization_24/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*N
shared_name?=feed_forward_sub_net_4/batch_normalization_24/moving_variance
�
Qfeed_forward_sub_net_4/batch_normalization_24/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_4/batch_normalization_24/moving_variance*
_output_shapes
:
*
dtype0
�
9feed_forward_sub_net_4/batch_normalization_25/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_4/batch_normalization_25/moving_mean
�
Mfeed_forward_sub_net_4/batch_normalization_25/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_4/batch_normalization_25/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_4/batch_normalization_25/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_4/batch_normalization_25/moving_variance
�
Qfeed_forward_sub_net_4/batch_normalization_25/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_4/batch_normalization_25/moving_variance*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_4/batch_normalization_26/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_4/batch_normalization_26/moving_mean
�
Mfeed_forward_sub_net_4/batch_normalization_26/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_4/batch_normalization_26/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_4/batch_normalization_26/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_4/batch_normalization_26/moving_variance
�
Qfeed_forward_sub_net_4/batch_normalization_26/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_4/batch_normalization_26/moving_variance*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_4/batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_4/batch_normalization_27/moving_mean
�
Mfeed_forward_sub_net_4/batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_4/batch_normalization_27/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_4/batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_4/batch_normalization_27/moving_variance
�
Qfeed_forward_sub_net_4/batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_4/batch_normalization_27/moving_variance*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_4/batch_normalization_28/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_4/batch_normalization_28/moving_mean
�
Mfeed_forward_sub_net_4/batch_normalization_28/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_4/batch_normalization_28/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_4/batch_normalization_28/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_4/batch_normalization_28/moving_variance
�
Qfeed_forward_sub_net_4/batch_normalization_28/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_4/batch_normalization_28/moving_variance*
_output_shapes
:*
dtype0
�
&feed_forward_sub_net_4/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&feed_forward_sub_net_4/dense_20/kernel
�
:feed_forward_sub_net_4/dense_20/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_4/dense_20/kernel*
_output_shapes

:
*
dtype0
�
&feed_forward_sub_net_4/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_4/dense_21/kernel
�
:feed_forward_sub_net_4/dense_21/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_4/dense_21/kernel*
_output_shapes

:*
dtype0
�
&feed_forward_sub_net_4/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_4/dense_22/kernel
�
:feed_forward_sub_net_4/dense_22/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_4/dense_22/kernel*
_output_shapes

:*
dtype0
�
&feed_forward_sub_net_4/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_4/dense_23/kernel
�
:feed_forward_sub_net_4/dense_23/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_4/dense_23/kernel*
_output_shapes

:*
dtype0
�
&feed_forward_sub_net_4/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&feed_forward_sub_net_4/dense_24/kernel
�
:feed_forward_sub_net_4/dense_24/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_4/dense_24/kernel*
_output_shapes

:
*
dtype0
�
$feed_forward_sub_net_4/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$feed_forward_sub_net_4/dense_24/bias
�
8feed_forward_sub_net_4/dense_24/bias/Read/ReadVariableOpReadVariableOp$feed_forward_sub_net_4/dense_24/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
�:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�:
value�:B�9 B�9
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
om
VARIABLE_VALUE3feed_forward_sub_net_4/batch_normalization_24/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_4/batch_normalization_24/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_4/batch_normalization_25/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_4/batch_normalization_25/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_4/batch_normalization_26/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_4/batch_normalization_26/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_4/batch_normalization_27/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_4/batch_normalization_27/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_4/batch_normalization_28/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_4/batch_normalization_28/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_4/batch_normalization_24/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_4/batch_normalization_24/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_4/batch_normalization_25/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_4/batch_normalization_25/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_4/batch_normalization_26/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_4/batch_normalization_26/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_4/batch_normalization_27/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_4/batch_normalization_27/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_4/batch_normalization_28/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_4/batch_normalization_28/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_4/dense_20/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_4/dense_21/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_4/dense_22/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_4/dense_23/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_4/dense_24/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$feed_forward_sub_net_4/dense_24/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_19feed_forward_sub_net_4/batch_normalization_24/moving_mean=feed_forward_sub_net_4/batch_normalization_24/moving_variance2feed_forward_sub_net_4/batch_normalization_24/beta3feed_forward_sub_net_4/batch_normalization_24/gamma&feed_forward_sub_net_4/dense_20/kernel9feed_forward_sub_net_4/batch_normalization_25/moving_mean=feed_forward_sub_net_4/batch_normalization_25/moving_variance2feed_forward_sub_net_4/batch_normalization_25/beta3feed_forward_sub_net_4/batch_normalization_25/gamma&feed_forward_sub_net_4/dense_21/kernel9feed_forward_sub_net_4/batch_normalization_26/moving_mean=feed_forward_sub_net_4/batch_normalization_26/moving_variance2feed_forward_sub_net_4/batch_normalization_26/beta3feed_forward_sub_net_4/batch_normalization_26/gamma&feed_forward_sub_net_4/dense_22/kernel9feed_forward_sub_net_4/batch_normalization_27/moving_mean=feed_forward_sub_net_4/batch_normalization_27/moving_variance2feed_forward_sub_net_4/batch_normalization_27/beta3feed_forward_sub_net_4/batch_normalization_27/gamma&feed_forward_sub_net_4/dense_23/kernel9feed_forward_sub_net_4/batch_normalization_28/moving_mean=feed_forward_sub_net_4/batch_normalization_28/moving_variance2feed_forward_sub_net_4/batch_normalization_28/beta3feed_forward_sub_net_4/batch_normalization_28/gamma&feed_forward_sub_net_4/dense_24/kernel$feed_forward_sub_net_4/dense_24/bias*&
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
$__inference_signature_wrapper_316543
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameGfeed_forward_sub_net_4/batch_normalization_24/gamma/Read/ReadVariableOpFfeed_forward_sub_net_4/batch_normalization_24/beta/Read/ReadVariableOpGfeed_forward_sub_net_4/batch_normalization_25/gamma/Read/ReadVariableOpFfeed_forward_sub_net_4/batch_normalization_25/beta/Read/ReadVariableOpGfeed_forward_sub_net_4/batch_normalization_26/gamma/Read/ReadVariableOpFfeed_forward_sub_net_4/batch_normalization_26/beta/Read/ReadVariableOpGfeed_forward_sub_net_4/batch_normalization_27/gamma/Read/ReadVariableOpFfeed_forward_sub_net_4/batch_normalization_27/beta/Read/ReadVariableOpGfeed_forward_sub_net_4/batch_normalization_28/gamma/Read/ReadVariableOpFfeed_forward_sub_net_4/batch_normalization_28/beta/Read/ReadVariableOpMfeed_forward_sub_net_4/batch_normalization_24/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_4/batch_normalization_24/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_4/batch_normalization_25/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_4/batch_normalization_25/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_4/batch_normalization_26/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_4/batch_normalization_26/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_4/batch_normalization_27/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_4/batch_normalization_27/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_4/batch_normalization_28/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_4/batch_normalization_28/moving_variance/Read/ReadVariableOp:feed_forward_sub_net_4/dense_20/kernel/Read/ReadVariableOp:feed_forward_sub_net_4/dense_21/kernel/Read/ReadVariableOp:feed_forward_sub_net_4/dense_22/kernel/Read/ReadVariableOp:feed_forward_sub_net_4/dense_23/kernel/Read/ReadVariableOp:feed_forward_sub_net_4/dense_24/kernel/Read/ReadVariableOp8feed_forward_sub_net_4/dense_24/bias/Read/ReadVariableOpConst*'
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
__inference__traced_save_317941
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename3feed_forward_sub_net_4/batch_normalization_24/gamma2feed_forward_sub_net_4/batch_normalization_24/beta3feed_forward_sub_net_4/batch_normalization_25/gamma2feed_forward_sub_net_4/batch_normalization_25/beta3feed_forward_sub_net_4/batch_normalization_26/gamma2feed_forward_sub_net_4/batch_normalization_26/beta3feed_forward_sub_net_4/batch_normalization_27/gamma2feed_forward_sub_net_4/batch_normalization_27/beta3feed_forward_sub_net_4/batch_normalization_28/gamma2feed_forward_sub_net_4/batch_normalization_28/beta9feed_forward_sub_net_4/batch_normalization_24/moving_mean=feed_forward_sub_net_4/batch_normalization_24/moving_variance9feed_forward_sub_net_4/batch_normalization_25/moving_mean=feed_forward_sub_net_4/batch_normalization_25/moving_variance9feed_forward_sub_net_4/batch_normalization_26/moving_mean=feed_forward_sub_net_4/batch_normalization_26/moving_variance9feed_forward_sub_net_4/batch_normalization_27/moving_mean=feed_forward_sub_net_4/batch_normalization_27/moving_variance9feed_forward_sub_net_4/batch_normalization_28/moving_mean=feed_forward_sub_net_4/batch_normalization_28/moving_variance&feed_forward_sub_net_4/dense_20/kernel&feed_forward_sub_net_4/dense_21/kernel&feed_forward_sub_net_4/dense_22/kernel&feed_forward_sub_net_4/dense_23/kernel&feed_forward_sub_net_4/dense_24/kernel$feed_forward_sub_net_4/dense_24/bias*&
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
"__inference__traced_restore_318029��
��
� 
!__inference__wrapped_model_315059
input_1X
Jfeed_forward_sub_net_4_batch_normalization_24_cast_readvariableop_resource:
Z
Lfeed_forward_sub_net_4_batch_normalization_24_cast_1_readvariableop_resource:
Z
Lfeed_forward_sub_net_4_batch_normalization_24_cast_2_readvariableop_resource:
Z
Lfeed_forward_sub_net_4_batch_normalization_24_cast_3_readvariableop_resource:
P
>feed_forward_sub_net_4_dense_20_matmul_readvariableop_resource:
X
Jfeed_forward_sub_net_4_batch_normalization_25_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_4_batch_normalization_25_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_4_batch_normalization_25_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_4_batch_normalization_25_cast_3_readvariableop_resource:P
>feed_forward_sub_net_4_dense_21_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_4_batch_normalization_26_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_4_batch_normalization_26_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_4_batch_normalization_26_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_4_batch_normalization_26_cast_3_readvariableop_resource:P
>feed_forward_sub_net_4_dense_22_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_4_batch_normalization_27_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_4_batch_normalization_27_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_4_batch_normalization_27_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_4_batch_normalization_27_cast_3_readvariableop_resource:P
>feed_forward_sub_net_4_dense_23_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_4_batch_normalization_28_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_4_batch_normalization_28_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_4_batch_normalization_28_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_4_batch_normalization_28_cast_3_readvariableop_resource:P
>feed_forward_sub_net_4_dense_24_matmul_readvariableop_resource:
M
?feed_forward_sub_net_4_dense_24_biasadd_readvariableop_resource:

identity��Afeed_forward_sub_net_4/batch_normalization_24/Cast/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_24/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_24/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_24/Cast_3/ReadVariableOp�Afeed_forward_sub_net_4/batch_normalization_25/Cast/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_25/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_25/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_25/Cast_3/ReadVariableOp�Afeed_forward_sub_net_4/batch_normalization_26/Cast/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_26/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_26/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_26/Cast_3/ReadVariableOp�Afeed_forward_sub_net_4/batch_normalization_27/Cast/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_27/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_27/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_27/Cast_3/ReadVariableOp�Afeed_forward_sub_net_4/batch_normalization_28/Cast/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_28/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_28/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_4/batch_normalization_28/Cast_3/ReadVariableOp�5feed_forward_sub_net_4/dense_20/MatMul/ReadVariableOp�5feed_forward_sub_net_4/dense_21/MatMul/ReadVariableOp�5feed_forward_sub_net_4/dense_22/MatMul/ReadVariableOp�5feed_forward_sub_net_4/dense_23/MatMul/ReadVariableOp�6feed_forward_sub_net_4/dense_24/BiasAdd/ReadVariableOp�5feed_forward_sub_net_4/dense_24/MatMul/ReadVariableOp�
Afeed_forward_sub_net_4/batch_normalization_24/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_4_batch_normalization_24_cast_readvariableop_resource*
_output_shapes
:
*
dtype02C
Afeed_forward_sub_net_4/batch_normalization_24/Cast/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_24/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_24_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_24/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_24/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_24_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_24/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_24/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_24_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_24/Cast_3/ReadVariableOp�
=feed_forward_sub_net_4/batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_4/batch_normalization_24/batchnorm/add/y�
;feed_forward_sub_net_4/batch_normalization_24/batchnorm/addAddV2Kfeed_forward_sub_net_4/batch_normalization_24/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_4/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_4/batch_normalization_24/batchnorm/add�
=feed_forward_sub_net_4/batch_normalization_24/batchnorm/RsqrtRsqrt?feed_forward_sub_net_4/batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_4/batch_normalization_24/batchnorm/Rsqrt�
;feed_forward_sub_net_4/batch_normalization_24/batchnorm/mulMulAfeed_forward_sub_net_4/batch_normalization_24/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_4/batch_normalization_24/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_4/batch_normalization_24/batchnorm/mul�
=feed_forward_sub_net_4/batch_normalization_24/batchnorm/mul_1Mulinput_1?feed_forward_sub_net_4/batch_normalization_24/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2?
=feed_forward_sub_net_4/batch_normalization_24/batchnorm/mul_1�
=feed_forward_sub_net_4/batch_normalization_24/batchnorm/mul_2MulIfeed_forward_sub_net_4/batch_normalization_24/Cast/ReadVariableOp:value:0?feed_forward_sub_net_4/batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_4/batch_normalization_24/batchnorm/mul_2�
;feed_forward_sub_net_4/batch_normalization_24/batchnorm/subSubKfeed_forward_sub_net_4/batch_normalization_24/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_4/batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_4/batch_normalization_24/batchnorm/sub�
=feed_forward_sub_net_4/batch_normalization_24/batchnorm/add_1AddV2Afeed_forward_sub_net_4/batch_normalization_24/batchnorm/mul_1:z:0?feed_forward_sub_net_4/batch_normalization_24/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2?
=feed_forward_sub_net_4/batch_normalization_24/batchnorm/add_1�
5feed_forward_sub_net_4/dense_20/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_4_dense_20_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5feed_forward_sub_net_4/dense_20/MatMul/ReadVariableOp�
&feed_forward_sub_net_4/dense_20/MatMulMatMulAfeed_forward_sub_net_4/batch_normalization_24/batchnorm/add_1:z:0=feed_forward_sub_net_4/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&feed_forward_sub_net_4/dense_20/MatMul�
Afeed_forward_sub_net_4/batch_normalization_25/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_4_batch_normalization_25_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_4/batch_normalization_25/Cast/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_25/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_25_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_25/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_25/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_25_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_25/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_25/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_25_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_25/Cast_3/ReadVariableOp�
=feed_forward_sub_net_4/batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_4/batch_normalization_25/batchnorm/add/y�
;feed_forward_sub_net_4/batch_normalization_25/batchnorm/addAddV2Kfeed_forward_sub_net_4/batch_normalization_25/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_4/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_4/batch_normalization_25/batchnorm/add�
=feed_forward_sub_net_4/batch_normalization_25/batchnorm/RsqrtRsqrt?feed_forward_sub_net_4/batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_4/batch_normalization_25/batchnorm/Rsqrt�
;feed_forward_sub_net_4/batch_normalization_25/batchnorm/mulMulAfeed_forward_sub_net_4/batch_normalization_25/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_4/batch_normalization_25/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_4/batch_normalization_25/batchnorm/mul�
=feed_forward_sub_net_4/batch_normalization_25/batchnorm/mul_1Mul0feed_forward_sub_net_4/dense_20/MatMul:product:0?feed_forward_sub_net_4/batch_normalization_25/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_4/batch_normalization_25/batchnorm/mul_1�
=feed_forward_sub_net_4/batch_normalization_25/batchnorm/mul_2MulIfeed_forward_sub_net_4/batch_normalization_25/Cast/ReadVariableOp:value:0?feed_forward_sub_net_4/batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_4/batch_normalization_25/batchnorm/mul_2�
;feed_forward_sub_net_4/batch_normalization_25/batchnorm/subSubKfeed_forward_sub_net_4/batch_normalization_25/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_4/batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_4/batch_normalization_25/batchnorm/sub�
=feed_forward_sub_net_4/batch_normalization_25/batchnorm/add_1AddV2Afeed_forward_sub_net_4/batch_normalization_25/batchnorm/mul_1:z:0?feed_forward_sub_net_4/batch_normalization_25/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_4/batch_normalization_25/batchnorm/add_1�
feed_forward_sub_net_4/ReluReluAfeed_forward_sub_net_4/batch_normalization_25/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_4/Relu�
5feed_forward_sub_net_4/dense_21/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_4_dense_21_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_4/dense_21/MatMul/ReadVariableOp�
&feed_forward_sub_net_4/dense_21/MatMulMatMul)feed_forward_sub_net_4/Relu:activations:0=feed_forward_sub_net_4/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&feed_forward_sub_net_4/dense_21/MatMul�
Afeed_forward_sub_net_4/batch_normalization_26/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_4_batch_normalization_26_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_4/batch_normalization_26/Cast/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_26/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_26/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_26/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_26_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_26/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_26/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_26_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_26/Cast_3/ReadVariableOp�
=feed_forward_sub_net_4/batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_4/batch_normalization_26/batchnorm/add/y�
;feed_forward_sub_net_4/batch_normalization_26/batchnorm/addAddV2Kfeed_forward_sub_net_4/batch_normalization_26/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_4/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_4/batch_normalization_26/batchnorm/add�
=feed_forward_sub_net_4/batch_normalization_26/batchnorm/RsqrtRsqrt?feed_forward_sub_net_4/batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_4/batch_normalization_26/batchnorm/Rsqrt�
;feed_forward_sub_net_4/batch_normalization_26/batchnorm/mulMulAfeed_forward_sub_net_4/batch_normalization_26/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_4/batch_normalization_26/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_4/batch_normalization_26/batchnorm/mul�
=feed_forward_sub_net_4/batch_normalization_26/batchnorm/mul_1Mul0feed_forward_sub_net_4/dense_21/MatMul:product:0?feed_forward_sub_net_4/batch_normalization_26/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_4/batch_normalization_26/batchnorm/mul_1�
=feed_forward_sub_net_4/batch_normalization_26/batchnorm/mul_2MulIfeed_forward_sub_net_4/batch_normalization_26/Cast/ReadVariableOp:value:0?feed_forward_sub_net_4/batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_4/batch_normalization_26/batchnorm/mul_2�
;feed_forward_sub_net_4/batch_normalization_26/batchnorm/subSubKfeed_forward_sub_net_4/batch_normalization_26/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_4/batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_4/batch_normalization_26/batchnorm/sub�
=feed_forward_sub_net_4/batch_normalization_26/batchnorm/add_1AddV2Afeed_forward_sub_net_4/batch_normalization_26/batchnorm/mul_1:z:0?feed_forward_sub_net_4/batch_normalization_26/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_4/batch_normalization_26/batchnorm/add_1�
feed_forward_sub_net_4/Relu_1ReluAfeed_forward_sub_net_4/batch_normalization_26/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_4/Relu_1�
5feed_forward_sub_net_4/dense_22/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_4_dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_4/dense_22/MatMul/ReadVariableOp�
&feed_forward_sub_net_4/dense_22/MatMulMatMul+feed_forward_sub_net_4/Relu_1:activations:0=feed_forward_sub_net_4/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&feed_forward_sub_net_4/dense_22/MatMul�
Afeed_forward_sub_net_4/batch_normalization_27/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_4_batch_normalization_27_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_4/batch_normalization_27/Cast/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_27/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_27_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_27/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_27/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_27_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_27/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_27/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_27_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_27/Cast_3/ReadVariableOp�
=feed_forward_sub_net_4/batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_4/batch_normalization_27/batchnorm/add/y�
;feed_forward_sub_net_4/batch_normalization_27/batchnorm/addAddV2Kfeed_forward_sub_net_4/batch_normalization_27/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_4/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_4/batch_normalization_27/batchnorm/add�
=feed_forward_sub_net_4/batch_normalization_27/batchnorm/RsqrtRsqrt?feed_forward_sub_net_4/batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_4/batch_normalization_27/batchnorm/Rsqrt�
;feed_forward_sub_net_4/batch_normalization_27/batchnorm/mulMulAfeed_forward_sub_net_4/batch_normalization_27/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_4/batch_normalization_27/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_4/batch_normalization_27/batchnorm/mul�
=feed_forward_sub_net_4/batch_normalization_27/batchnorm/mul_1Mul0feed_forward_sub_net_4/dense_22/MatMul:product:0?feed_forward_sub_net_4/batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_4/batch_normalization_27/batchnorm/mul_1�
=feed_forward_sub_net_4/batch_normalization_27/batchnorm/mul_2MulIfeed_forward_sub_net_4/batch_normalization_27/Cast/ReadVariableOp:value:0?feed_forward_sub_net_4/batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_4/batch_normalization_27/batchnorm/mul_2�
;feed_forward_sub_net_4/batch_normalization_27/batchnorm/subSubKfeed_forward_sub_net_4/batch_normalization_27/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_4/batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_4/batch_normalization_27/batchnorm/sub�
=feed_forward_sub_net_4/batch_normalization_27/batchnorm/add_1AddV2Afeed_forward_sub_net_4/batch_normalization_27/batchnorm/mul_1:z:0?feed_forward_sub_net_4/batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_4/batch_normalization_27/batchnorm/add_1�
feed_forward_sub_net_4/Relu_2ReluAfeed_forward_sub_net_4/batch_normalization_27/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_4/Relu_2�
5feed_forward_sub_net_4/dense_23/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_4_dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_4/dense_23/MatMul/ReadVariableOp�
&feed_forward_sub_net_4/dense_23/MatMulMatMul+feed_forward_sub_net_4/Relu_2:activations:0=feed_forward_sub_net_4/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&feed_forward_sub_net_4/dense_23/MatMul�
Afeed_forward_sub_net_4/batch_normalization_28/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_4_batch_normalization_28_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_4/batch_normalization_28/Cast/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_28/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_28_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_28/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_28/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_28_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_28/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_4/batch_normalization_28/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_4_batch_normalization_28_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_4/batch_normalization_28/Cast_3/ReadVariableOp�
=feed_forward_sub_net_4/batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_4/batch_normalization_28/batchnorm/add/y�
;feed_forward_sub_net_4/batch_normalization_28/batchnorm/addAddV2Kfeed_forward_sub_net_4/batch_normalization_28/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_4/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_4/batch_normalization_28/batchnorm/add�
=feed_forward_sub_net_4/batch_normalization_28/batchnorm/RsqrtRsqrt?feed_forward_sub_net_4/batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_4/batch_normalization_28/batchnorm/Rsqrt�
;feed_forward_sub_net_4/batch_normalization_28/batchnorm/mulMulAfeed_forward_sub_net_4/batch_normalization_28/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_4/batch_normalization_28/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_4/batch_normalization_28/batchnorm/mul�
=feed_forward_sub_net_4/batch_normalization_28/batchnorm/mul_1Mul0feed_forward_sub_net_4/dense_23/MatMul:product:0?feed_forward_sub_net_4/batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_4/batch_normalization_28/batchnorm/mul_1�
=feed_forward_sub_net_4/batch_normalization_28/batchnorm/mul_2MulIfeed_forward_sub_net_4/batch_normalization_28/Cast/ReadVariableOp:value:0?feed_forward_sub_net_4/batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_4/batch_normalization_28/batchnorm/mul_2�
;feed_forward_sub_net_4/batch_normalization_28/batchnorm/subSubKfeed_forward_sub_net_4/batch_normalization_28/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_4/batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_4/batch_normalization_28/batchnorm/sub�
=feed_forward_sub_net_4/batch_normalization_28/batchnorm/add_1AddV2Afeed_forward_sub_net_4/batch_normalization_28/batchnorm/mul_1:z:0?feed_forward_sub_net_4/batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_4/batch_normalization_28/batchnorm/add_1�
feed_forward_sub_net_4/Relu_3ReluAfeed_forward_sub_net_4/batch_normalization_28/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_4/Relu_3�
5feed_forward_sub_net_4/dense_24/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_4_dense_24_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5feed_forward_sub_net_4/dense_24/MatMul/ReadVariableOp�
&feed_forward_sub_net_4/dense_24/MatMulMatMul+feed_forward_sub_net_4/Relu_3:activations:0=feed_forward_sub_net_4/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2(
&feed_forward_sub_net_4/dense_24/MatMul�
6feed_forward_sub_net_4/dense_24/BiasAdd/ReadVariableOpReadVariableOp?feed_forward_sub_net_4_dense_24_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype028
6feed_forward_sub_net_4/dense_24/BiasAdd/ReadVariableOp�
'feed_forward_sub_net_4/dense_24/BiasAddBiasAdd0feed_forward_sub_net_4/dense_24/MatMul:product:0>feed_forward_sub_net_4/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2)
'feed_forward_sub_net_4/dense_24/BiasAdd�
IdentityIdentity0feed_forward_sub_net_4/dense_24/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOpB^feed_forward_sub_net_4/batch_normalization_24/Cast/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_24/Cast_1/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_24/Cast_2/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_24/Cast_3/ReadVariableOpB^feed_forward_sub_net_4/batch_normalization_25/Cast/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_25/Cast_1/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_25/Cast_2/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_25/Cast_3/ReadVariableOpB^feed_forward_sub_net_4/batch_normalization_26/Cast/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_26/Cast_1/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_26/Cast_2/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_26/Cast_3/ReadVariableOpB^feed_forward_sub_net_4/batch_normalization_27/Cast/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_27/Cast_1/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_27/Cast_2/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_27/Cast_3/ReadVariableOpB^feed_forward_sub_net_4/batch_normalization_28/Cast/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_28/Cast_1/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_28/Cast_2/ReadVariableOpD^feed_forward_sub_net_4/batch_normalization_28/Cast_3/ReadVariableOp6^feed_forward_sub_net_4/dense_20/MatMul/ReadVariableOp6^feed_forward_sub_net_4/dense_21/MatMul/ReadVariableOp6^feed_forward_sub_net_4/dense_22/MatMul/ReadVariableOp6^feed_forward_sub_net_4/dense_23/MatMul/ReadVariableOp7^feed_forward_sub_net_4/dense_24/BiasAdd/ReadVariableOp6^feed_forward_sub_net_4/dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2�
Afeed_forward_sub_net_4/batch_normalization_24/Cast/ReadVariableOpAfeed_forward_sub_net_4/batch_normalization_24/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_24/Cast_1/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_24/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_24/Cast_2/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_24/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_24/Cast_3/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_24/Cast_3/ReadVariableOp2�
Afeed_forward_sub_net_4/batch_normalization_25/Cast/ReadVariableOpAfeed_forward_sub_net_4/batch_normalization_25/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_25/Cast_1/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_25/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_25/Cast_2/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_25/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_25/Cast_3/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_25/Cast_3/ReadVariableOp2�
Afeed_forward_sub_net_4/batch_normalization_26/Cast/ReadVariableOpAfeed_forward_sub_net_4/batch_normalization_26/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_26/Cast_1/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_26/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_26/Cast_2/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_26/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_26/Cast_3/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_26/Cast_3/ReadVariableOp2�
Afeed_forward_sub_net_4/batch_normalization_27/Cast/ReadVariableOpAfeed_forward_sub_net_4/batch_normalization_27/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_27/Cast_1/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_27/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_27/Cast_2/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_27/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_27/Cast_3/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_27/Cast_3/ReadVariableOp2�
Afeed_forward_sub_net_4/batch_normalization_28/Cast/ReadVariableOpAfeed_forward_sub_net_4/batch_normalization_28/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_28/Cast_1/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_28/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_28/Cast_2/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_28/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_4/batch_normalization_28/Cast_3/ReadVariableOpCfeed_forward_sub_net_4/batch_normalization_28/Cast_3/ReadVariableOp2n
5feed_forward_sub_net_4/dense_20/MatMul/ReadVariableOp5feed_forward_sub_net_4/dense_20/MatMul/ReadVariableOp2n
5feed_forward_sub_net_4/dense_21/MatMul/ReadVariableOp5feed_forward_sub_net_4/dense_21/MatMul/ReadVariableOp2n
5feed_forward_sub_net_4/dense_22/MatMul/ReadVariableOp5feed_forward_sub_net_4/dense_22/MatMul/ReadVariableOp2n
5feed_forward_sub_net_4/dense_23/MatMul/ReadVariableOp5feed_forward_sub_net_4/dense_23/MatMul/ReadVariableOp2p
6feed_forward_sub_net_4/dense_24/BiasAdd/ReadVariableOp6feed_forward_sub_net_4/dense_24/BiasAdd/ReadVariableOp2n
5feed_forward_sub_net_4/dense_24/MatMul/ReadVariableOp5feed_forward_sub_net_4/dense_24/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
�
�
7__inference_feed_forward_sub_net_4_layer_call_fn_317355
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
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_3162322
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
�+
�
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_315809

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
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_315083

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
�E
�
__inference__traced_save_317941
file_prefixR
Nsavev2_feed_forward_sub_net_4_batch_normalization_24_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_4_batch_normalization_24_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_4_batch_normalization_25_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_4_batch_normalization_25_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_4_batch_normalization_26_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_4_batch_normalization_26_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_4_batch_normalization_27_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_4_batch_normalization_27_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_4_batch_normalization_28_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_4_batch_normalization_28_beta_read_readvariableopX
Tsavev2_feed_forward_sub_net_4_batch_normalization_24_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_4_batch_normalization_24_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_4_batch_normalization_25_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_4_batch_normalization_25_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_4_batch_normalization_26_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_4_batch_normalization_26_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_4_batch_normalization_27_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_4_batch_normalization_27_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_4_batch_normalization_28_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_4_batch_normalization_28_moving_variance_read_readvariableopE
Asavev2_feed_forward_sub_net_4_dense_20_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_4_dense_21_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_4_dense_22_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_4_dense_23_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_4_dense_24_kernel_read_readvariableopC
?savev2_feed_forward_sub_net_4_dense_24_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Nsavev2_feed_forward_sub_net_4_batch_normalization_24_gamma_read_readvariableopMsavev2_feed_forward_sub_net_4_batch_normalization_24_beta_read_readvariableopNsavev2_feed_forward_sub_net_4_batch_normalization_25_gamma_read_readvariableopMsavev2_feed_forward_sub_net_4_batch_normalization_25_beta_read_readvariableopNsavev2_feed_forward_sub_net_4_batch_normalization_26_gamma_read_readvariableopMsavev2_feed_forward_sub_net_4_batch_normalization_26_beta_read_readvariableopNsavev2_feed_forward_sub_net_4_batch_normalization_27_gamma_read_readvariableopMsavev2_feed_forward_sub_net_4_batch_normalization_27_beta_read_readvariableopNsavev2_feed_forward_sub_net_4_batch_normalization_28_gamma_read_readvariableopMsavev2_feed_forward_sub_net_4_batch_normalization_28_beta_read_readvariableopTsavev2_feed_forward_sub_net_4_batch_normalization_24_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_4_batch_normalization_24_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_4_batch_normalization_25_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_4_batch_normalization_25_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_4_batch_normalization_26_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_4_batch_normalization_26_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_4_batch_normalization_27_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_4_batch_normalization_27_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_4_batch_normalization_28_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_4_batch_normalization_28_moving_variance_read_readvariableopAsavev2_feed_forward_sub_net_4_dense_20_kernel_read_readvariableopAsavev2_feed_forward_sub_net_4_dense_21_kernel_read_readvariableopAsavev2_feed_forward_sub_net_4_dense_22_kernel_read_readvariableopAsavev2_feed_forward_sub_net_4_dense_23_kernel_read_readvariableopAsavev2_feed_forward_sub_net_4_dense_24_kernel_read_readvariableop?savev2_feed_forward_sub_net_4_dense_24_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�
�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_317375

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
�+
�
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_317575

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
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_315311

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
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_315643

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
�D
�
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_316006
x+
batch_normalization_24_315896:
+
batch_normalization_24_315898:
+
batch_normalization_24_315900:
+
batch_normalization_24_315902:
!
dense_20_315913:
+
batch_normalization_25_315916:+
batch_normalization_25_315918:+
batch_normalization_25_315920:+
batch_normalization_25_315922:!
dense_21_315934:+
batch_normalization_26_315937:+
batch_normalization_26_315939:+
batch_normalization_26_315941:+
batch_normalization_26_315943:!
dense_22_315955:+
batch_normalization_27_315958:+
batch_normalization_27_315960:+
batch_normalization_27_315962:+
batch_normalization_27_315964:!
dense_23_315976:+
batch_normalization_28_315979:+
batch_normalization_28_315981:+
batch_normalization_28_315983:+
batch_normalization_28_315985:!
dense_24_316000:

dense_24_316002:

identity��.batch_normalization_24/StatefulPartitionedCall�.batch_normalization_25/StatefulPartitionedCall�.batch_normalization_26/StatefulPartitionedCall�.batch_normalization_27/StatefulPartitionedCall�.batch_normalization_28/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_24_315896batch_normalization_24_315898batch_normalization_24_315900batch_normalization_24_315902*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_31508320
.batch_normalization_24/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0dense_20_315913*
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
GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_3159122"
 dense_20/StatefulPartitionedCall�
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_25_315916batch_normalization_25_315918batch_normalization_25_315920batch_normalization_25_315922*
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
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_31524920
.batch_normalization_25/StatefulPartitionedCall
ReluRelu7batch_normalization_25/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu�
 dense_21/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_21_315934*
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
GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_3159332"
 dense_21/StatefulPartitionedCall�
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_26_315937batch_normalization_26_315939batch_normalization_26_315941batch_normalization_26_315943*
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
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_31541520
.batch_normalization_26/StatefulPartitionedCall�
Relu_1Relu7batch_normalization_26/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_1�
 dense_22/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_22_315955*
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
GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_3159542"
 dense_22/StatefulPartitionedCall�
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_27_315958batch_normalization_27_315960batch_normalization_27_315962batch_normalization_27_315964*
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
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_31558120
.batch_normalization_27/StatefulPartitionedCall�
Relu_2Relu7batch_normalization_27/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_2�
 dense_23/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_23_315976*
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
GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_3159752"
 dense_23/StatefulPartitionedCall�
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0batch_normalization_28_315979batch_normalization_28_315981batch_normalization_28_315983batch_normalization_28_315985*
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
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_31574720
.batch_normalization_28/StatefulPartitionedCall�
Relu_3Relu7batch_normalization_28/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_3�
 dense_24/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_24_316000dense_24_316002*
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
GPU 2J 8� *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_3159992"
 dense_24/StatefulPartitionedCall�
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:J F
'
_output_shapes
:���������


_user_specified_namex
�
�
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_315581

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
�
�
D__inference_dense_23_layer_call_and_return_conditional_losses_315975

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
7__inference_batch_normalization_27_layer_call_fn_317670

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
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3155812
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
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_317411

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
7__inference_batch_normalization_24_layer_call_fn_317424

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3150832
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
�
�
$__inference_signature_wrapper_316543
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
!__inference__wrapped_model_3150592
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
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_317457

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
�
�
D__inference_dense_20_layer_call_and_return_conditional_losses_317772

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
�
�
D__inference_dense_23_layer_call_and_return_conditional_losses_317814

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
D__inference_dense_21_layer_call_and_return_conditional_losses_317786

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
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_317539

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
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_317657

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
��
�
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_316649
xA
3batch_normalization_24_cast_readvariableop_resource:
C
5batch_normalization_24_cast_1_readvariableop_resource:
C
5batch_normalization_24_cast_2_readvariableop_resource:
C
5batch_normalization_24_cast_3_readvariableop_resource:
9
'dense_20_matmul_readvariableop_resource:
A
3batch_normalization_25_cast_readvariableop_resource:C
5batch_normalization_25_cast_1_readvariableop_resource:C
5batch_normalization_25_cast_2_readvariableop_resource:C
5batch_normalization_25_cast_3_readvariableop_resource:9
'dense_21_matmul_readvariableop_resource:A
3batch_normalization_26_cast_readvariableop_resource:C
5batch_normalization_26_cast_1_readvariableop_resource:C
5batch_normalization_26_cast_2_readvariableop_resource:C
5batch_normalization_26_cast_3_readvariableop_resource:9
'dense_22_matmul_readvariableop_resource:A
3batch_normalization_27_cast_readvariableop_resource:C
5batch_normalization_27_cast_1_readvariableop_resource:C
5batch_normalization_27_cast_2_readvariableop_resource:C
5batch_normalization_27_cast_3_readvariableop_resource:9
'dense_23_matmul_readvariableop_resource:A
3batch_normalization_28_cast_readvariableop_resource:C
5batch_normalization_28_cast_1_readvariableop_resource:C
5batch_normalization_28_cast_2_readvariableop_resource:C
5batch_normalization_28_cast_3_readvariableop_resource:9
'dense_24_matmul_readvariableop_resource:
6
(dense_24_biasadd_readvariableop_resource:

identity��*batch_normalization_24/Cast/ReadVariableOp�,batch_normalization_24/Cast_1/ReadVariableOp�,batch_normalization_24/Cast_2/ReadVariableOp�,batch_normalization_24/Cast_3/ReadVariableOp�*batch_normalization_25/Cast/ReadVariableOp�,batch_normalization_25/Cast_1/ReadVariableOp�,batch_normalization_25/Cast_2/ReadVariableOp�,batch_normalization_25/Cast_3/ReadVariableOp�*batch_normalization_26/Cast/ReadVariableOp�,batch_normalization_26/Cast_1/ReadVariableOp�,batch_normalization_26/Cast_2/ReadVariableOp�,batch_normalization_26/Cast_3/ReadVariableOp�*batch_normalization_27/Cast/ReadVariableOp�,batch_normalization_27/Cast_1/ReadVariableOp�,batch_normalization_27/Cast_2/ReadVariableOp�,batch_normalization_27/Cast_3/ReadVariableOp�*batch_normalization_28/Cast/ReadVariableOp�,batch_normalization_28/Cast_1/ReadVariableOp�,batch_normalization_28/Cast_2/ReadVariableOp�,batch_normalization_28/Cast_3/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�
*batch_normalization_24/Cast/ReadVariableOpReadVariableOp3batch_normalization_24_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_24/Cast/ReadVariableOp�
,batch_normalization_24/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_24_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_24/Cast_1/ReadVariableOp�
,batch_normalization_24/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_24_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_24/Cast_2/ReadVariableOp�
,batch_normalization_24/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_24_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_24/Cast_3/ReadVariableOp�
&batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_24/batchnorm/add/y�
$batch_normalization_24/batchnorm/addAddV24batch_normalization_24/Cast_1/ReadVariableOp:value:0/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_24/batchnorm/add�
&batch_normalization_24/batchnorm/RsqrtRsqrt(batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_24/batchnorm/Rsqrt�
$batch_normalization_24/batchnorm/mulMul*batch_normalization_24/batchnorm/Rsqrt:y:04batch_normalization_24/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_24/batchnorm/mul�
&batch_normalization_24/batchnorm/mul_1Mulx(batch_normalization_24/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_24/batchnorm/mul_1�
&batch_normalization_24/batchnorm/mul_2Mul2batch_normalization_24/Cast/ReadVariableOp:value:0(batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_24/batchnorm/mul_2�
$batch_normalization_24/batchnorm/subSub4batch_normalization_24/Cast_2/ReadVariableOp:value:0*batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_24/batchnorm/sub�
&batch_normalization_24/batchnorm/add_1AddV2*batch_normalization_24/batchnorm/mul_1:z:0(batch_normalization_24/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_24/batchnorm/add_1�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul*batch_normalization_24/batchnorm/add_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
*batch_normalization_25/Cast/ReadVariableOpReadVariableOp3batch_normalization_25_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_25/Cast/ReadVariableOp�
,batch_normalization_25/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_25_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_25/Cast_1/ReadVariableOp�
,batch_normalization_25/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_25_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_25/Cast_2/ReadVariableOp�
,batch_normalization_25/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_25_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_25/Cast_3/ReadVariableOp�
&batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_25/batchnorm/add/y�
$batch_normalization_25/batchnorm/addAddV24batch_normalization_25/Cast_1/ReadVariableOp:value:0/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_25/batchnorm/add�
&batch_normalization_25/batchnorm/RsqrtRsqrt(batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_25/batchnorm/Rsqrt�
$batch_normalization_25/batchnorm/mulMul*batch_normalization_25/batchnorm/Rsqrt:y:04batch_normalization_25/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_25/batchnorm/mul�
&batch_normalization_25/batchnorm/mul_1Muldense_20/MatMul:product:0(batch_normalization_25/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_25/batchnorm/mul_1�
&batch_normalization_25/batchnorm/mul_2Mul2batch_normalization_25/Cast/ReadVariableOp:value:0(batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_25/batchnorm/mul_2�
$batch_normalization_25/batchnorm/subSub4batch_normalization_25/Cast_2/ReadVariableOp:value:0*batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_25/batchnorm/sub�
&batch_normalization_25/batchnorm/add_1AddV2*batch_normalization_25/batchnorm/mul_1:z:0(batch_normalization_25/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_25/batchnorm/add_1r
ReluRelu*batch_normalization_25/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_21/MatMul/ReadVariableOp�
dense_21/MatMulMatMulRelu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_21/MatMul�
*batch_normalization_26/Cast/ReadVariableOpReadVariableOp3batch_normalization_26_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_26/Cast/ReadVariableOp�
,batch_normalization_26/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_26/Cast_1/ReadVariableOp�
,batch_normalization_26/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_26_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_26/Cast_2/ReadVariableOp�
,batch_normalization_26/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_26_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_26/Cast_3/ReadVariableOp�
&batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_26/batchnorm/add/y�
$batch_normalization_26/batchnorm/addAddV24batch_normalization_26/Cast_1/ReadVariableOp:value:0/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_26/batchnorm/add�
&batch_normalization_26/batchnorm/RsqrtRsqrt(batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_26/batchnorm/Rsqrt�
$batch_normalization_26/batchnorm/mulMul*batch_normalization_26/batchnorm/Rsqrt:y:04batch_normalization_26/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_26/batchnorm/mul�
&batch_normalization_26/batchnorm/mul_1Muldense_21/MatMul:product:0(batch_normalization_26/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_26/batchnorm/mul_1�
&batch_normalization_26/batchnorm/mul_2Mul2batch_normalization_26/Cast/ReadVariableOp:value:0(batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_26/batchnorm/mul_2�
$batch_normalization_26/batchnorm/subSub4batch_normalization_26/Cast_2/ReadVariableOp:value:0*batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_26/batchnorm/sub�
&batch_normalization_26/batchnorm/add_1AddV2*batch_normalization_26/batchnorm/mul_1:z:0(batch_normalization_26/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_26/batchnorm/add_1v
Relu_1Relu*batch_normalization_26/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_22/MatMul/ReadVariableOp�
dense_22/MatMulMatMulRelu_1:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_22/MatMul�
*batch_normalization_27/Cast/ReadVariableOpReadVariableOp3batch_normalization_27_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_27/Cast/ReadVariableOp�
,batch_normalization_27/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_27_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_27/Cast_1/ReadVariableOp�
,batch_normalization_27/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_27_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_27/Cast_2/ReadVariableOp�
,batch_normalization_27/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_27_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_27/Cast_3/ReadVariableOp�
&batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_27/batchnorm/add/y�
$batch_normalization_27/batchnorm/addAddV24batch_normalization_27/Cast_1/ReadVariableOp:value:0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_27/batchnorm/add�
&batch_normalization_27/batchnorm/RsqrtRsqrt(batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_27/batchnorm/Rsqrt�
$batch_normalization_27/batchnorm/mulMul*batch_normalization_27/batchnorm/Rsqrt:y:04batch_normalization_27/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_27/batchnorm/mul�
&batch_normalization_27/batchnorm/mul_1Muldense_22/MatMul:product:0(batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_27/batchnorm/mul_1�
&batch_normalization_27/batchnorm/mul_2Mul2batch_normalization_27/Cast/ReadVariableOp:value:0(batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_27/batchnorm/mul_2�
$batch_normalization_27/batchnorm/subSub4batch_normalization_27/Cast_2/ReadVariableOp:value:0*batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_27/batchnorm/sub�
&batch_normalization_27/batchnorm/add_1AddV2*batch_normalization_27/batchnorm/mul_1:z:0(batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_27/batchnorm/add_1v
Relu_2Relu*batch_normalization_27/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_23/MatMul/ReadVariableOp�
dense_23/MatMulMatMulRelu_2:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_23/MatMul�
*batch_normalization_28/Cast/ReadVariableOpReadVariableOp3batch_normalization_28_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_28/Cast/ReadVariableOp�
,batch_normalization_28/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_28_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_28/Cast_1/ReadVariableOp�
,batch_normalization_28/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_28_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_28/Cast_2/ReadVariableOp�
,batch_normalization_28/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_28_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_28/Cast_3/ReadVariableOp�
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_28/batchnorm/add/y�
$batch_normalization_28/batchnorm/addAddV24batch_normalization_28/Cast_1/ReadVariableOp:value:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_28/batchnorm/add�
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_28/batchnorm/Rsqrt�
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:04batch_normalization_28/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_28/batchnorm/mul�
&batch_normalization_28/batchnorm/mul_1Muldense_23/MatMul:product:0(batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_28/batchnorm/mul_1�
&batch_normalization_28/batchnorm/mul_2Mul2batch_normalization_28/Cast/ReadVariableOp:value:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_28/batchnorm/mul_2�
$batch_normalization_28/batchnorm/subSub4batch_normalization_28/Cast_2/ReadVariableOp:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_28/batchnorm/sub�
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_28/batchnorm/add_1v
Relu_3Relu*batch_normalization_28/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_24/MatMul/ReadVariableOp�
dense_24/MatMulMatMulRelu_3:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_24/MatMul�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_24/BiasAdd/ReadVariableOp�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_24/BiasAddt
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�	
NoOpNoOp+^batch_normalization_24/Cast/ReadVariableOp-^batch_normalization_24/Cast_1/ReadVariableOp-^batch_normalization_24/Cast_2/ReadVariableOp-^batch_normalization_24/Cast_3/ReadVariableOp+^batch_normalization_25/Cast/ReadVariableOp-^batch_normalization_25/Cast_1/ReadVariableOp-^batch_normalization_25/Cast_2/ReadVariableOp-^batch_normalization_25/Cast_3/ReadVariableOp+^batch_normalization_26/Cast/ReadVariableOp-^batch_normalization_26/Cast_1/ReadVariableOp-^batch_normalization_26/Cast_2/ReadVariableOp-^batch_normalization_26/Cast_3/ReadVariableOp+^batch_normalization_27/Cast/ReadVariableOp-^batch_normalization_27/Cast_1/ReadVariableOp-^batch_normalization_27/Cast_2/ReadVariableOp-^batch_normalization_27/Cast_3/ReadVariableOp+^batch_normalization_28/Cast/ReadVariableOp-^batch_normalization_28/Cast_1/ReadVariableOp-^batch_normalization_28/Cast_2/ReadVariableOp-^batch_normalization_28/Cast_3/ReadVariableOp^dense_20/MatMul/ReadVariableOp^dense_21/MatMul/ReadVariableOp^dense_22/MatMul/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_24/Cast/ReadVariableOp*batch_normalization_24/Cast/ReadVariableOp2\
,batch_normalization_24/Cast_1/ReadVariableOp,batch_normalization_24/Cast_1/ReadVariableOp2\
,batch_normalization_24/Cast_2/ReadVariableOp,batch_normalization_24/Cast_2/ReadVariableOp2\
,batch_normalization_24/Cast_3/ReadVariableOp,batch_normalization_24/Cast_3/ReadVariableOp2X
*batch_normalization_25/Cast/ReadVariableOp*batch_normalization_25/Cast/ReadVariableOp2\
,batch_normalization_25/Cast_1/ReadVariableOp,batch_normalization_25/Cast_1/ReadVariableOp2\
,batch_normalization_25/Cast_2/ReadVariableOp,batch_normalization_25/Cast_2/ReadVariableOp2\
,batch_normalization_25/Cast_3/ReadVariableOp,batch_normalization_25/Cast_3/ReadVariableOp2X
*batch_normalization_26/Cast/ReadVariableOp*batch_normalization_26/Cast/ReadVariableOp2\
,batch_normalization_26/Cast_1/ReadVariableOp,batch_normalization_26/Cast_1/ReadVariableOp2\
,batch_normalization_26/Cast_2/ReadVariableOp,batch_normalization_26/Cast_2/ReadVariableOp2\
,batch_normalization_26/Cast_3/ReadVariableOp,batch_normalization_26/Cast_3/ReadVariableOp2X
*batch_normalization_27/Cast/ReadVariableOp*batch_normalization_27/Cast/ReadVariableOp2\
,batch_normalization_27/Cast_1/ReadVariableOp,batch_normalization_27/Cast_1/ReadVariableOp2\
,batch_normalization_27/Cast_2/ReadVariableOp,batch_normalization_27/Cast_2/ReadVariableOp2\
,batch_normalization_27/Cast_3/ReadVariableOp,batch_normalization_27/Cast_3/ReadVariableOp2X
*batch_normalization_28/Cast/ReadVariableOp*batch_normalization_28/Cast/ReadVariableOp2\
,batch_normalization_28/Cast_1/ReadVariableOp,batch_normalization_28/Cast_1/ReadVariableOp2\
,batch_normalization_28/Cast_2/ReadVariableOp,batch_normalization_28/Cast_2/ReadVariableOp2\
,batch_normalization_28/Cast_3/ReadVariableOp,batch_normalization_28/Cast_3/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������


_user_specified_namex
�+
�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_315145

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
�
}
)__inference_dense_21_layer_call_fn_317793

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
GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_3159332
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
�
�
7__inference_feed_forward_sub_net_4_layer_call_fn_317298
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
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_3162322
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
�

�
D__inference_dense_24_layer_call_and_return_conditional_losses_315999

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
�
�
7__inference_feed_forward_sub_net_4_layer_call_fn_317241
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
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_3160062
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
�D
�
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_316232
x+
batch_normalization_24_316165:
+
batch_normalization_24_316167:
+
batch_normalization_24_316169:
+
batch_normalization_24_316171:
!
dense_20_316174:
+
batch_normalization_25_316177:+
batch_normalization_25_316179:+
batch_normalization_25_316181:+
batch_normalization_25_316183:!
dense_21_316187:+
batch_normalization_26_316190:+
batch_normalization_26_316192:+
batch_normalization_26_316194:+
batch_normalization_26_316196:!
dense_22_316200:+
batch_normalization_27_316203:+
batch_normalization_27_316205:+
batch_normalization_27_316207:+
batch_normalization_27_316209:!
dense_23_316213:+
batch_normalization_28_316216:+
batch_normalization_28_316218:+
batch_normalization_28_316220:+
batch_normalization_28_316222:!
dense_24_316226:

dense_24_316228:

identity��.batch_normalization_24/StatefulPartitionedCall�.batch_normalization_25/StatefulPartitionedCall�.batch_normalization_26/StatefulPartitionedCall�.batch_normalization_27/StatefulPartitionedCall�.batch_normalization_28/StatefulPartitionedCall� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�
.batch_normalization_24/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_24_316165batch_normalization_24_316167batch_normalization_24_316169batch_normalization_24_316171*
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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_31514520
.batch_normalization_24/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_24/StatefulPartitionedCall:output:0dense_20_316174*
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
GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_3159122"
 dense_20/StatefulPartitionedCall�
.batch_normalization_25/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0batch_normalization_25_316177batch_normalization_25_316179batch_normalization_25_316181batch_normalization_25_316183*
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
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_31531120
.batch_normalization_25/StatefulPartitionedCall
ReluRelu7batch_normalization_25/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu�
 dense_21/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_21_316187*
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
GPU 2J 8� *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_3159332"
 dense_21/StatefulPartitionedCall�
.batch_normalization_26/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0batch_normalization_26_316190batch_normalization_26_316192batch_normalization_26_316194batch_normalization_26_316196*
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
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_31547720
.batch_normalization_26/StatefulPartitionedCall�
Relu_1Relu7batch_normalization_26/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_1�
 dense_22/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_22_316200*
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
GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_3159542"
 dense_22/StatefulPartitionedCall�
.batch_normalization_27/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_27_316203batch_normalization_27_316205batch_normalization_27_316207batch_normalization_27_316209*
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
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_31564320
.batch_normalization_27/StatefulPartitionedCall�
Relu_2Relu7batch_normalization_27/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_2�
 dense_23/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_23_316213*
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
GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_3159752"
 dense_23/StatefulPartitionedCall�
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0batch_normalization_28_316216batch_normalization_28_316218batch_normalization_28_316220batch_normalization_28_316222*
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
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_31580920
.batch_normalization_28/StatefulPartitionedCall�
Relu_3Relu7batch_normalization_28/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_3�
 dense_24/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_24_316226dense_24_316228*
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
GPU 2J 8� *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_3159992"
 dense_24/StatefulPartitionedCall�
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp/^batch_normalization_24/StatefulPartitionedCall/^batch_normalization_25/StatefulPartitionedCall/^batch_normalization_26/StatefulPartitionedCall/^batch_normalization_27/StatefulPartitionedCall/^batch_normalization_28/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_24/StatefulPartitionedCall.batch_normalization_24/StatefulPartitionedCall2`
.batch_normalization_25/StatefulPartitionedCall.batch_normalization_25/StatefulPartitionedCall2`
.batch_normalization_26/StatefulPartitionedCall.batch_normalization_26/StatefulPartitionedCall2`
.batch_normalization_27/StatefulPartitionedCall.batch_normalization_27/StatefulPartitionedCall2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:J F
'
_output_shapes
:���������


_user_specified_namex
�
�
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_315249

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
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_317621

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
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_315747

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
�
�
D__inference_dense_22_layer_call_and_return_conditional_losses_315954

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
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_316835
xL
>batch_normalization_24_assignmovingavg_readvariableop_resource:
N
@batch_normalization_24_assignmovingavg_1_readvariableop_resource:
A
3batch_normalization_24_cast_readvariableop_resource:
C
5batch_normalization_24_cast_1_readvariableop_resource:
9
'dense_20_matmul_readvariableop_resource:
L
>batch_normalization_25_assignmovingavg_readvariableop_resource:N
@batch_normalization_25_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_25_cast_readvariableop_resource:C
5batch_normalization_25_cast_1_readvariableop_resource:9
'dense_21_matmul_readvariableop_resource:L
>batch_normalization_26_assignmovingavg_readvariableop_resource:N
@batch_normalization_26_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_26_cast_readvariableop_resource:C
5batch_normalization_26_cast_1_readvariableop_resource:9
'dense_22_matmul_readvariableop_resource:L
>batch_normalization_27_assignmovingavg_readvariableop_resource:N
@batch_normalization_27_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_27_cast_readvariableop_resource:C
5batch_normalization_27_cast_1_readvariableop_resource:9
'dense_23_matmul_readvariableop_resource:L
>batch_normalization_28_assignmovingavg_readvariableop_resource:N
@batch_normalization_28_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_28_cast_readvariableop_resource:C
5batch_normalization_28_cast_1_readvariableop_resource:9
'dense_24_matmul_readvariableop_resource:
6
(dense_24_biasadd_readvariableop_resource:

identity��&batch_normalization_24/AssignMovingAvg�5batch_normalization_24/AssignMovingAvg/ReadVariableOp�(batch_normalization_24/AssignMovingAvg_1�7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_24/Cast/ReadVariableOp�,batch_normalization_24/Cast_1/ReadVariableOp�&batch_normalization_25/AssignMovingAvg�5batch_normalization_25/AssignMovingAvg/ReadVariableOp�(batch_normalization_25/AssignMovingAvg_1�7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_25/Cast/ReadVariableOp�,batch_normalization_25/Cast_1/ReadVariableOp�&batch_normalization_26/AssignMovingAvg�5batch_normalization_26/AssignMovingAvg/ReadVariableOp�(batch_normalization_26/AssignMovingAvg_1�7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_26/Cast/ReadVariableOp�,batch_normalization_26/Cast_1/ReadVariableOp�&batch_normalization_27/AssignMovingAvg�5batch_normalization_27/AssignMovingAvg/ReadVariableOp�(batch_normalization_27/AssignMovingAvg_1�7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_27/Cast/ReadVariableOp�,batch_normalization_27/Cast_1/ReadVariableOp�&batch_normalization_28/AssignMovingAvg�5batch_normalization_28/AssignMovingAvg/ReadVariableOp�(batch_normalization_28/AssignMovingAvg_1�7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_28/Cast/ReadVariableOp�,batch_normalization_28/Cast_1/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�
5batch_normalization_24/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_24/moments/mean/reduction_indices�
#batch_normalization_24/moments/meanMeanx>batch_normalization_24/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_24/moments/mean�
+batch_normalization_24/moments/StopGradientStopGradient,batch_normalization_24/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_24/moments/StopGradient�
0batch_normalization_24/moments/SquaredDifferenceSquaredDifferencex4batch_normalization_24/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������
22
0batch_normalization_24/moments/SquaredDifference�
9batch_normalization_24/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_24/moments/variance/reduction_indices�
'batch_normalization_24/moments/varianceMean4batch_normalization_24/moments/SquaredDifference:z:0Bbatch_normalization_24/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_24/moments/variance�
&batch_normalization_24/moments/SqueezeSqueeze,batch_normalization_24/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_24/moments/Squeeze�
(batch_normalization_24/moments/Squeeze_1Squeeze0batch_normalization_24/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_24/moments/Squeeze_1�
,batch_normalization_24/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_24/AssignMovingAvg/decay�
+batch_normalization_24/AssignMovingAvg/CastCast5batch_normalization_24/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_24/AssignMovingAvg/Cast�
5batch_normalization_24/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_24_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_24/AssignMovingAvg/ReadVariableOp�
*batch_normalization_24/AssignMovingAvg/subSub=batch_normalization_24/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_24/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_24/AssignMovingAvg/sub�
*batch_normalization_24/AssignMovingAvg/mulMul.batch_normalization_24/AssignMovingAvg/sub:z:0/batch_normalization_24/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_24/AssignMovingAvg/mul�
&batch_normalization_24/AssignMovingAvgAssignSubVariableOp>batch_normalization_24_assignmovingavg_readvariableop_resource.batch_normalization_24/AssignMovingAvg/mul:z:06^batch_normalization_24/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_24/AssignMovingAvg�
.batch_normalization_24/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_24/AssignMovingAvg_1/decay�
-batch_normalization_24/AssignMovingAvg_1/CastCast7batch_normalization_24/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_24/AssignMovingAvg_1/Cast�
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_24_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_24/AssignMovingAvg_1/subSub?batch_normalization_24/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_24/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_24/AssignMovingAvg_1/sub�
,batch_normalization_24/AssignMovingAvg_1/mulMul0batch_normalization_24/AssignMovingAvg_1/sub:z:01batch_normalization_24/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_24/AssignMovingAvg_1/mul�
(batch_normalization_24/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_24_assignmovingavg_1_readvariableop_resource0batch_normalization_24/AssignMovingAvg_1/mul:z:08^batch_normalization_24/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_24/AssignMovingAvg_1�
*batch_normalization_24/Cast/ReadVariableOpReadVariableOp3batch_normalization_24_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_24/Cast/ReadVariableOp�
,batch_normalization_24/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_24_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_24/Cast_1/ReadVariableOp�
&batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_24/batchnorm/add/y�
$batch_normalization_24/batchnorm/addAddV21batch_normalization_24/moments/Squeeze_1:output:0/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_24/batchnorm/add�
&batch_normalization_24/batchnorm/RsqrtRsqrt(batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_24/batchnorm/Rsqrt�
$batch_normalization_24/batchnorm/mulMul*batch_normalization_24/batchnorm/Rsqrt:y:04batch_normalization_24/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_24/batchnorm/mul�
&batch_normalization_24/batchnorm/mul_1Mulx(batch_normalization_24/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_24/batchnorm/mul_1�
&batch_normalization_24/batchnorm/mul_2Mul/batch_normalization_24/moments/Squeeze:output:0(batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_24/batchnorm/mul_2�
$batch_normalization_24/batchnorm/subSub2batch_normalization_24/Cast/ReadVariableOp:value:0*batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_24/batchnorm/sub�
&batch_normalization_24/batchnorm/add_1AddV2*batch_normalization_24/batchnorm/mul_1:z:0(batch_normalization_24/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_24/batchnorm/add_1�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul*batch_normalization_24/batchnorm/add_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
5batch_normalization_25/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_25/moments/mean/reduction_indices�
#batch_normalization_25/moments/meanMeandense_20/MatMul:product:0>batch_normalization_25/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_25/moments/mean�
+batch_normalization_25/moments/StopGradientStopGradient,batch_normalization_25/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_25/moments/StopGradient�
0batch_normalization_25/moments/SquaredDifferenceSquaredDifferencedense_20/MatMul:product:04batch_normalization_25/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_25/moments/SquaredDifference�
9batch_normalization_25/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_25/moments/variance/reduction_indices�
'batch_normalization_25/moments/varianceMean4batch_normalization_25/moments/SquaredDifference:z:0Bbatch_normalization_25/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_25/moments/variance�
&batch_normalization_25/moments/SqueezeSqueeze,batch_normalization_25/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_25/moments/Squeeze�
(batch_normalization_25/moments/Squeeze_1Squeeze0batch_normalization_25/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_25/moments/Squeeze_1�
,batch_normalization_25/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_25/AssignMovingAvg/decay�
+batch_normalization_25/AssignMovingAvg/CastCast5batch_normalization_25/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_25/AssignMovingAvg/Cast�
5batch_normalization_25/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_25_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_25/AssignMovingAvg/ReadVariableOp�
*batch_normalization_25/AssignMovingAvg/subSub=batch_normalization_25/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_25/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_25/AssignMovingAvg/sub�
*batch_normalization_25/AssignMovingAvg/mulMul.batch_normalization_25/AssignMovingAvg/sub:z:0/batch_normalization_25/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_25/AssignMovingAvg/mul�
&batch_normalization_25/AssignMovingAvgAssignSubVariableOp>batch_normalization_25_assignmovingavg_readvariableop_resource.batch_normalization_25/AssignMovingAvg/mul:z:06^batch_normalization_25/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_25/AssignMovingAvg�
.batch_normalization_25/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_25/AssignMovingAvg_1/decay�
-batch_normalization_25/AssignMovingAvg_1/CastCast7batch_normalization_25/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_25/AssignMovingAvg_1/Cast�
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_25_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_25/AssignMovingAvg_1/subSub?batch_normalization_25/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_25/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_25/AssignMovingAvg_1/sub�
,batch_normalization_25/AssignMovingAvg_1/mulMul0batch_normalization_25/AssignMovingAvg_1/sub:z:01batch_normalization_25/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_25/AssignMovingAvg_1/mul�
(batch_normalization_25/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_25_assignmovingavg_1_readvariableop_resource0batch_normalization_25/AssignMovingAvg_1/mul:z:08^batch_normalization_25/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_25/AssignMovingAvg_1�
*batch_normalization_25/Cast/ReadVariableOpReadVariableOp3batch_normalization_25_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_25/Cast/ReadVariableOp�
,batch_normalization_25/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_25_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_25/Cast_1/ReadVariableOp�
&batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_25/batchnorm/add/y�
$batch_normalization_25/batchnorm/addAddV21batch_normalization_25/moments/Squeeze_1:output:0/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_25/batchnorm/add�
&batch_normalization_25/batchnorm/RsqrtRsqrt(batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_25/batchnorm/Rsqrt�
$batch_normalization_25/batchnorm/mulMul*batch_normalization_25/batchnorm/Rsqrt:y:04batch_normalization_25/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_25/batchnorm/mul�
&batch_normalization_25/batchnorm/mul_1Muldense_20/MatMul:product:0(batch_normalization_25/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_25/batchnorm/mul_1�
&batch_normalization_25/batchnorm/mul_2Mul/batch_normalization_25/moments/Squeeze:output:0(batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_25/batchnorm/mul_2�
$batch_normalization_25/batchnorm/subSub2batch_normalization_25/Cast/ReadVariableOp:value:0*batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_25/batchnorm/sub�
&batch_normalization_25/batchnorm/add_1AddV2*batch_normalization_25/batchnorm/mul_1:z:0(batch_normalization_25/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_25/batchnorm/add_1r
ReluRelu*batch_normalization_25/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_21/MatMul/ReadVariableOp�
dense_21/MatMulMatMulRelu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_21/MatMul�
5batch_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_26/moments/mean/reduction_indices�
#batch_normalization_26/moments/meanMeandense_21/MatMul:product:0>batch_normalization_26/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_26/moments/mean�
+batch_normalization_26/moments/StopGradientStopGradient,batch_normalization_26/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_26/moments/StopGradient�
0batch_normalization_26/moments/SquaredDifferenceSquaredDifferencedense_21/MatMul:product:04batch_normalization_26/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_26/moments/SquaredDifference�
9batch_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_26/moments/variance/reduction_indices�
'batch_normalization_26/moments/varianceMean4batch_normalization_26/moments/SquaredDifference:z:0Bbatch_normalization_26/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_26/moments/variance�
&batch_normalization_26/moments/SqueezeSqueeze,batch_normalization_26/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_26/moments/Squeeze�
(batch_normalization_26/moments/Squeeze_1Squeeze0batch_normalization_26/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_26/moments/Squeeze_1�
,batch_normalization_26/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_26/AssignMovingAvg/decay�
+batch_normalization_26/AssignMovingAvg/CastCast5batch_normalization_26/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_26/AssignMovingAvg/Cast�
5batch_normalization_26/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_26_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_26/AssignMovingAvg/ReadVariableOp�
*batch_normalization_26/AssignMovingAvg/subSub=batch_normalization_26/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_26/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_26/AssignMovingAvg/sub�
*batch_normalization_26/AssignMovingAvg/mulMul.batch_normalization_26/AssignMovingAvg/sub:z:0/batch_normalization_26/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_26/AssignMovingAvg/mul�
&batch_normalization_26/AssignMovingAvgAssignSubVariableOp>batch_normalization_26_assignmovingavg_readvariableop_resource.batch_normalization_26/AssignMovingAvg/mul:z:06^batch_normalization_26/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_26/AssignMovingAvg�
.batch_normalization_26/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_26/AssignMovingAvg_1/decay�
-batch_normalization_26/AssignMovingAvg_1/CastCast7batch_normalization_26/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_26/AssignMovingAvg_1/Cast�
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_26_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_26/AssignMovingAvg_1/subSub?batch_normalization_26/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_26/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_26/AssignMovingAvg_1/sub�
,batch_normalization_26/AssignMovingAvg_1/mulMul0batch_normalization_26/AssignMovingAvg_1/sub:z:01batch_normalization_26/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_26/AssignMovingAvg_1/mul�
(batch_normalization_26/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_26_assignmovingavg_1_readvariableop_resource0batch_normalization_26/AssignMovingAvg_1/mul:z:08^batch_normalization_26/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_26/AssignMovingAvg_1�
*batch_normalization_26/Cast/ReadVariableOpReadVariableOp3batch_normalization_26_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_26/Cast/ReadVariableOp�
,batch_normalization_26/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_26/Cast_1/ReadVariableOp�
&batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_26/batchnorm/add/y�
$batch_normalization_26/batchnorm/addAddV21batch_normalization_26/moments/Squeeze_1:output:0/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_26/batchnorm/add�
&batch_normalization_26/batchnorm/RsqrtRsqrt(batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_26/batchnorm/Rsqrt�
$batch_normalization_26/batchnorm/mulMul*batch_normalization_26/batchnorm/Rsqrt:y:04batch_normalization_26/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_26/batchnorm/mul�
&batch_normalization_26/batchnorm/mul_1Muldense_21/MatMul:product:0(batch_normalization_26/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_26/batchnorm/mul_1�
&batch_normalization_26/batchnorm/mul_2Mul/batch_normalization_26/moments/Squeeze:output:0(batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_26/batchnorm/mul_2�
$batch_normalization_26/batchnorm/subSub2batch_normalization_26/Cast/ReadVariableOp:value:0*batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_26/batchnorm/sub�
&batch_normalization_26/batchnorm/add_1AddV2*batch_normalization_26/batchnorm/mul_1:z:0(batch_normalization_26/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_26/batchnorm/add_1v
Relu_1Relu*batch_normalization_26/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_22/MatMul/ReadVariableOp�
dense_22/MatMulMatMulRelu_1:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_22/MatMul�
5batch_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_27/moments/mean/reduction_indices�
#batch_normalization_27/moments/meanMeandense_22/MatMul:product:0>batch_normalization_27/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_27/moments/mean�
+batch_normalization_27/moments/StopGradientStopGradient,batch_normalization_27/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_27/moments/StopGradient�
0batch_normalization_27/moments/SquaredDifferenceSquaredDifferencedense_22/MatMul:product:04batch_normalization_27/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_27/moments/SquaredDifference�
9batch_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_27/moments/variance/reduction_indices�
'batch_normalization_27/moments/varianceMean4batch_normalization_27/moments/SquaredDifference:z:0Bbatch_normalization_27/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_27/moments/variance�
&batch_normalization_27/moments/SqueezeSqueeze,batch_normalization_27/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_27/moments/Squeeze�
(batch_normalization_27/moments/Squeeze_1Squeeze0batch_normalization_27/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_27/moments/Squeeze_1�
,batch_normalization_27/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_27/AssignMovingAvg/decay�
+batch_normalization_27/AssignMovingAvg/CastCast5batch_normalization_27/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_27/AssignMovingAvg/Cast�
5batch_normalization_27/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_27_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_27/AssignMovingAvg/ReadVariableOp�
*batch_normalization_27/AssignMovingAvg/subSub=batch_normalization_27/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_27/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_27/AssignMovingAvg/sub�
*batch_normalization_27/AssignMovingAvg/mulMul.batch_normalization_27/AssignMovingAvg/sub:z:0/batch_normalization_27/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_27/AssignMovingAvg/mul�
&batch_normalization_27/AssignMovingAvgAssignSubVariableOp>batch_normalization_27_assignmovingavg_readvariableop_resource.batch_normalization_27/AssignMovingAvg/mul:z:06^batch_normalization_27/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_27/AssignMovingAvg�
.batch_normalization_27/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_27/AssignMovingAvg_1/decay�
-batch_normalization_27/AssignMovingAvg_1/CastCast7batch_normalization_27/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_27/AssignMovingAvg_1/Cast�
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_27_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_27/AssignMovingAvg_1/subSub?batch_normalization_27/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_27/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_27/AssignMovingAvg_1/sub�
,batch_normalization_27/AssignMovingAvg_1/mulMul0batch_normalization_27/AssignMovingAvg_1/sub:z:01batch_normalization_27/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_27/AssignMovingAvg_1/mul�
(batch_normalization_27/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_27_assignmovingavg_1_readvariableop_resource0batch_normalization_27/AssignMovingAvg_1/mul:z:08^batch_normalization_27/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_27/AssignMovingAvg_1�
*batch_normalization_27/Cast/ReadVariableOpReadVariableOp3batch_normalization_27_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_27/Cast/ReadVariableOp�
,batch_normalization_27/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_27_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_27/Cast_1/ReadVariableOp�
&batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_27/batchnorm/add/y�
$batch_normalization_27/batchnorm/addAddV21batch_normalization_27/moments/Squeeze_1:output:0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_27/batchnorm/add�
&batch_normalization_27/batchnorm/RsqrtRsqrt(batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_27/batchnorm/Rsqrt�
$batch_normalization_27/batchnorm/mulMul*batch_normalization_27/batchnorm/Rsqrt:y:04batch_normalization_27/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_27/batchnorm/mul�
&batch_normalization_27/batchnorm/mul_1Muldense_22/MatMul:product:0(batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_27/batchnorm/mul_1�
&batch_normalization_27/batchnorm/mul_2Mul/batch_normalization_27/moments/Squeeze:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_27/batchnorm/mul_2�
$batch_normalization_27/batchnorm/subSub2batch_normalization_27/Cast/ReadVariableOp:value:0*batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_27/batchnorm/sub�
&batch_normalization_27/batchnorm/add_1AddV2*batch_normalization_27/batchnorm/mul_1:z:0(batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_27/batchnorm/add_1v
Relu_2Relu*batch_normalization_27/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_23/MatMul/ReadVariableOp�
dense_23/MatMulMatMulRelu_2:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_23/MatMul�
5batch_normalization_28/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_28/moments/mean/reduction_indices�
#batch_normalization_28/moments/meanMeandense_23/MatMul:product:0>batch_normalization_28/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_28/moments/mean�
+batch_normalization_28/moments/StopGradientStopGradient,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_28/moments/StopGradient�
0batch_normalization_28/moments/SquaredDifferenceSquaredDifferencedense_23/MatMul:product:04batch_normalization_28/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_28/moments/SquaredDifference�
9batch_normalization_28/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_28/moments/variance/reduction_indices�
'batch_normalization_28/moments/varianceMean4batch_normalization_28/moments/SquaredDifference:z:0Bbatch_normalization_28/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_28/moments/variance�
&batch_normalization_28/moments/SqueezeSqueeze,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_28/moments/Squeeze�
(batch_normalization_28/moments/Squeeze_1Squeeze0batch_normalization_28/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_28/moments/Squeeze_1�
,batch_normalization_28/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_28/AssignMovingAvg/decay�
+batch_normalization_28/AssignMovingAvg/CastCast5batch_normalization_28/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_28/AssignMovingAvg/Cast�
5batch_normalization_28/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_28_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_28/AssignMovingAvg/ReadVariableOp�
*batch_normalization_28/AssignMovingAvg/subSub=batch_normalization_28/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_28/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_28/AssignMovingAvg/sub�
*batch_normalization_28/AssignMovingAvg/mulMul.batch_normalization_28/AssignMovingAvg/sub:z:0/batch_normalization_28/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_28/AssignMovingAvg/mul�
&batch_normalization_28/AssignMovingAvgAssignSubVariableOp>batch_normalization_28_assignmovingavg_readvariableop_resource.batch_normalization_28/AssignMovingAvg/mul:z:06^batch_normalization_28/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_28/AssignMovingAvg�
.batch_normalization_28/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_28/AssignMovingAvg_1/decay�
-batch_normalization_28/AssignMovingAvg_1/CastCast7batch_normalization_28/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_28/AssignMovingAvg_1/Cast�
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_28/AssignMovingAvg_1/subSub?batch_normalization_28/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_28/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_28/AssignMovingAvg_1/sub�
,batch_normalization_28/AssignMovingAvg_1/mulMul0batch_normalization_28/AssignMovingAvg_1/sub:z:01batch_normalization_28/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_28/AssignMovingAvg_1/mul�
(batch_normalization_28/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource0batch_normalization_28/AssignMovingAvg_1/mul:z:08^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_28/AssignMovingAvg_1�
*batch_normalization_28/Cast/ReadVariableOpReadVariableOp3batch_normalization_28_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_28/Cast/ReadVariableOp�
,batch_normalization_28/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_28_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_28/Cast_1/ReadVariableOp�
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_28/batchnorm/add/y�
$batch_normalization_28/batchnorm/addAddV21batch_normalization_28/moments/Squeeze_1:output:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_28/batchnorm/add�
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_28/batchnorm/Rsqrt�
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:04batch_normalization_28/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_28/batchnorm/mul�
&batch_normalization_28/batchnorm/mul_1Muldense_23/MatMul:product:0(batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_28/batchnorm/mul_1�
&batch_normalization_28/batchnorm/mul_2Mul/batch_normalization_28/moments/Squeeze:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_28/batchnorm/mul_2�
$batch_normalization_28/batchnorm/subSub2batch_normalization_28/Cast/ReadVariableOp:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_28/batchnorm/sub�
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_28/batchnorm/add_1v
Relu_3Relu*batch_normalization_28/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_24/MatMul/ReadVariableOp�
dense_24/MatMulMatMulRelu_3:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_24/MatMul�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_24/BiasAdd/ReadVariableOp�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_24/BiasAddt
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp'^batch_normalization_24/AssignMovingAvg6^batch_normalization_24/AssignMovingAvg/ReadVariableOp)^batch_normalization_24/AssignMovingAvg_18^batch_normalization_24/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_24/Cast/ReadVariableOp-^batch_normalization_24/Cast_1/ReadVariableOp'^batch_normalization_25/AssignMovingAvg6^batch_normalization_25/AssignMovingAvg/ReadVariableOp)^batch_normalization_25/AssignMovingAvg_18^batch_normalization_25/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_25/Cast/ReadVariableOp-^batch_normalization_25/Cast_1/ReadVariableOp'^batch_normalization_26/AssignMovingAvg6^batch_normalization_26/AssignMovingAvg/ReadVariableOp)^batch_normalization_26/AssignMovingAvg_18^batch_normalization_26/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_26/Cast/ReadVariableOp-^batch_normalization_26/Cast_1/ReadVariableOp'^batch_normalization_27/AssignMovingAvg6^batch_normalization_27/AssignMovingAvg/ReadVariableOp)^batch_normalization_27/AssignMovingAvg_18^batch_normalization_27/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_27/Cast/ReadVariableOp-^batch_normalization_27/Cast_1/ReadVariableOp'^batch_normalization_28/AssignMovingAvg6^batch_normalization_28/AssignMovingAvg/ReadVariableOp)^batch_normalization_28/AssignMovingAvg_18^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_28/Cast/ReadVariableOp-^batch_normalization_28/Cast_1/ReadVariableOp^dense_20/MatMul/ReadVariableOp^dense_21/MatMul/ReadVariableOp^dense_22/MatMul/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_24/AssignMovingAvg&batch_normalization_24/AssignMovingAvg2n
5batch_normalization_24/AssignMovingAvg/ReadVariableOp5batch_normalization_24/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_24/AssignMovingAvg_1(batch_normalization_24/AssignMovingAvg_12r
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_24/Cast/ReadVariableOp*batch_normalization_24/Cast/ReadVariableOp2\
,batch_normalization_24/Cast_1/ReadVariableOp,batch_normalization_24/Cast_1/ReadVariableOp2P
&batch_normalization_25/AssignMovingAvg&batch_normalization_25/AssignMovingAvg2n
5batch_normalization_25/AssignMovingAvg/ReadVariableOp5batch_normalization_25/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_25/AssignMovingAvg_1(batch_normalization_25/AssignMovingAvg_12r
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_25/Cast/ReadVariableOp*batch_normalization_25/Cast/ReadVariableOp2\
,batch_normalization_25/Cast_1/ReadVariableOp,batch_normalization_25/Cast_1/ReadVariableOp2P
&batch_normalization_26/AssignMovingAvg&batch_normalization_26/AssignMovingAvg2n
5batch_normalization_26/AssignMovingAvg/ReadVariableOp5batch_normalization_26/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_26/AssignMovingAvg_1(batch_normalization_26/AssignMovingAvg_12r
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_26/Cast/ReadVariableOp*batch_normalization_26/Cast/ReadVariableOp2\
,batch_normalization_26/Cast_1/ReadVariableOp,batch_normalization_26/Cast_1/ReadVariableOp2P
&batch_normalization_27/AssignMovingAvg&batch_normalization_27/AssignMovingAvg2n
5batch_normalization_27/AssignMovingAvg/ReadVariableOp5batch_normalization_27/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_27/AssignMovingAvg_1(batch_normalization_27/AssignMovingAvg_12r
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_27/Cast/ReadVariableOp*batch_normalization_27/Cast/ReadVariableOp2\
,batch_normalization_27/Cast_1/ReadVariableOp,batch_normalization_27/Cast_1/ReadVariableOp2P
&batch_normalization_28/AssignMovingAvg&batch_normalization_28/AssignMovingAvg2n
5batch_normalization_28/AssignMovingAvg/ReadVariableOp5batch_normalization_28/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_28/AssignMovingAvg_1(batch_normalization_28/AssignMovingAvg_12r
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_28/Cast/ReadVariableOp*batch_normalization_28/Cast/ReadVariableOp2\
,batch_normalization_28/Cast_1/ReadVariableOp,batch_normalization_28/Cast_1/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������


_user_specified_namex
�+
�
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_315477

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
��
�
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_316941
input_1A
3batch_normalization_24_cast_readvariableop_resource:
C
5batch_normalization_24_cast_1_readvariableop_resource:
C
5batch_normalization_24_cast_2_readvariableop_resource:
C
5batch_normalization_24_cast_3_readvariableop_resource:
9
'dense_20_matmul_readvariableop_resource:
A
3batch_normalization_25_cast_readvariableop_resource:C
5batch_normalization_25_cast_1_readvariableop_resource:C
5batch_normalization_25_cast_2_readvariableop_resource:C
5batch_normalization_25_cast_3_readvariableop_resource:9
'dense_21_matmul_readvariableop_resource:A
3batch_normalization_26_cast_readvariableop_resource:C
5batch_normalization_26_cast_1_readvariableop_resource:C
5batch_normalization_26_cast_2_readvariableop_resource:C
5batch_normalization_26_cast_3_readvariableop_resource:9
'dense_22_matmul_readvariableop_resource:A
3batch_normalization_27_cast_readvariableop_resource:C
5batch_normalization_27_cast_1_readvariableop_resource:C
5batch_normalization_27_cast_2_readvariableop_resource:C
5batch_normalization_27_cast_3_readvariableop_resource:9
'dense_23_matmul_readvariableop_resource:A
3batch_normalization_28_cast_readvariableop_resource:C
5batch_normalization_28_cast_1_readvariableop_resource:C
5batch_normalization_28_cast_2_readvariableop_resource:C
5batch_normalization_28_cast_3_readvariableop_resource:9
'dense_24_matmul_readvariableop_resource:
6
(dense_24_biasadd_readvariableop_resource:

identity��*batch_normalization_24/Cast/ReadVariableOp�,batch_normalization_24/Cast_1/ReadVariableOp�,batch_normalization_24/Cast_2/ReadVariableOp�,batch_normalization_24/Cast_3/ReadVariableOp�*batch_normalization_25/Cast/ReadVariableOp�,batch_normalization_25/Cast_1/ReadVariableOp�,batch_normalization_25/Cast_2/ReadVariableOp�,batch_normalization_25/Cast_3/ReadVariableOp�*batch_normalization_26/Cast/ReadVariableOp�,batch_normalization_26/Cast_1/ReadVariableOp�,batch_normalization_26/Cast_2/ReadVariableOp�,batch_normalization_26/Cast_3/ReadVariableOp�*batch_normalization_27/Cast/ReadVariableOp�,batch_normalization_27/Cast_1/ReadVariableOp�,batch_normalization_27/Cast_2/ReadVariableOp�,batch_normalization_27/Cast_3/ReadVariableOp�*batch_normalization_28/Cast/ReadVariableOp�,batch_normalization_28/Cast_1/ReadVariableOp�,batch_normalization_28/Cast_2/ReadVariableOp�,batch_normalization_28/Cast_3/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�
*batch_normalization_24/Cast/ReadVariableOpReadVariableOp3batch_normalization_24_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_24/Cast/ReadVariableOp�
,batch_normalization_24/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_24_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_24/Cast_1/ReadVariableOp�
,batch_normalization_24/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_24_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_24/Cast_2/ReadVariableOp�
,batch_normalization_24/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_24_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_24/Cast_3/ReadVariableOp�
&batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_24/batchnorm/add/y�
$batch_normalization_24/batchnorm/addAddV24batch_normalization_24/Cast_1/ReadVariableOp:value:0/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_24/batchnorm/add�
&batch_normalization_24/batchnorm/RsqrtRsqrt(batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_24/batchnorm/Rsqrt�
$batch_normalization_24/batchnorm/mulMul*batch_normalization_24/batchnorm/Rsqrt:y:04batch_normalization_24/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_24/batchnorm/mul�
&batch_normalization_24/batchnorm/mul_1Mulinput_1(batch_normalization_24/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_24/batchnorm/mul_1�
&batch_normalization_24/batchnorm/mul_2Mul2batch_normalization_24/Cast/ReadVariableOp:value:0(batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_24/batchnorm/mul_2�
$batch_normalization_24/batchnorm/subSub4batch_normalization_24/Cast_2/ReadVariableOp:value:0*batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_24/batchnorm/sub�
&batch_normalization_24/batchnorm/add_1AddV2*batch_normalization_24/batchnorm/mul_1:z:0(batch_normalization_24/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_24/batchnorm/add_1�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul*batch_normalization_24/batchnorm/add_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
*batch_normalization_25/Cast/ReadVariableOpReadVariableOp3batch_normalization_25_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_25/Cast/ReadVariableOp�
,batch_normalization_25/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_25_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_25/Cast_1/ReadVariableOp�
,batch_normalization_25/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_25_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_25/Cast_2/ReadVariableOp�
,batch_normalization_25/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_25_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_25/Cast_3/ReadVariableOp�
&batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_25/batchnorm/add/y�
$batch_normalization_25/batchnorm/addAddV24batch_normalization_25/Cast_1/ReadVariableOp:value:0/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_25/batchnorm/add�
&batch_normalization_25/batchnorm/RsqrtRsqrt(batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_25/batchnorm/Rsqrt�
$batch_normalization_25/batchnorm/mulMul*batch_normalization_25/batchnorm/Rsqrt:y:04batch_normalization_25/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_25/batchnorm/mul�
&batch_normalization_25/batchnorm/mul_1Muldense_20/MatMul:product:0(batch_normalization_25/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_25/batchnorm/mul_1�
&batch_normalization_25/batchnorm/mul_2Mul2batch_normalization_25/Cast/ReadVariableOp:value:0(batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_25/batchnorm/mul_2�
$batch_normalization_25/batchnorm/subSub4batch_normalization_25/Cast_2/ReadVariableOp:value:0*batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_25/batchnorm/sub�
&batch_normalization_25/batchnorm/add_1AddV2*batch_normalization_25/batchnorm/mul_1:z:0(batch_normalization_25/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_25/batchnorm/add_1r
ReluRelu*batch_normalization_25/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_21/MatMul/ReadVariableOp�
dense_21/MatMulMatMulRelu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_21/MatMul�
*batch_normalization_26/Cast/ReadVariableOpReadVariableOp3batch_normalization_26_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_26/Cast/ReadVariableOp�
,batch_normalization_26/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_26/Cast_1/ReadVariableOp�
,batch_normalization_26/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_26_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_26/Cast_2/ReadVariableOp�
,batch_normalization_26/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_26_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_26/Cast_3/ReadVariableOp�
&batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_26/batchnorm/add/y�
$batch_normalization_26/batchnorm/addAddV24batch_normalization_26/Cast_1/ReadVariableOp:value:0/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_26/batchnorm/add�
&batch_normalization_26/batchnorm/RsqrtRsqrt(batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_26/batchnorm/Rsqrt�
$batch_normalization_26/batchnorm/mulMul*batch_normalization_26/batchnorm/Rsqrt:y:04batch_normalization_26/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_26/batchnorm/mul�
&batch_normalization_26/batchnorm/mul_1Muldense_21/MatMul:product:0(batch_normalization_26/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_26/batchnorm/mul_1�
&batch_normalization_26/batchnorm/mul_2Mul2batch_normalization_26/Cast/ReadVariableOp:value:0(batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_26/batchnorm/mul_2�
$batch_normalization_26/batchnorm/subSub4batch_normalization_26/Cast_2/ReadVariableOp:value:0*batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_26/batchnorm/sub�
&batch_normalization_26/batchnorm/add_1AddV2*batch_normalization_26/batchnorm/mul_1:z:0(batch_normalization_26/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_26/batchnorm/add_1v
Relu_1Relu*batch_normalization_26/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_22/MatMul/ReadVariableOp�
dense_22/MatMulMatMulRelu_1:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_22/MatMul�
*batch_normalization_27/Cast/ReadVariableOpReadVariableOp3batch_normalization_27_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_27/Cast/ReadVariableOp�
,batch_normalization_27/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_27_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_27/Cast_1/ReadVariableOp�
,batch_normalization_27/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_27_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_27/Cast_2/ReadVariableOp�
,batch_normalization_27/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_27_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_27/Cast_3/ReadVariableOp�
&batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_27/batchnorm/add/y�
$batch_normalization_27/batchnorm/addAddV24batch_normalization_27/Cast_1/ReadVariableOp:value:0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_27/batchnorm/add�
&batch_normalization_27/batchnorm/RsqrtRsqrt(batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_27/batchnorm/Rsqrt�
$batch_normalization_27/batchnorm/mulMul*batch_normalization_27/batchnorm/Rsqrt:y:04batch_normalization_27/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_27/batchnorm/mul�
&batch_normalization_27/batchnorm/mul_1Muldense_22/MatMul:product:0(batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_27/batchnorm/mul_1�
&batch_normalization_27/batchnorm/mul_2Mul2batch_normalization_27/Cast/ReadVariableOp:value:0(batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_27/batchnorm/mul_2�
$batch_normalization_27/batchnorm/subSub4batch_normalization_27/Cast_2/ReadVariableOp:value:0*batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_27/batchnorm/sub�
&batch_normalization_27/batchnorm/add_1AddV2*batch_normalization_27/batchnorm/mul_1:z:0(batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_27/batchnorm/add_1v
Relu_2Relu*batch_normalization_27/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_23/MatMul/ReadVariableOp�
dense_23/MatMulMatMulRelu_2:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_23/MatMul�
*batch_normalization_28/Cast/ReadVariableOpReadVariableOp3batch_normalization_28_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_28/Cast/ReadVariableOp�
,batch_normalization_28/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_28_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_28/Cast_1/ReadVariableOp�
,batch_normalization_28/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_28_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_28/Cast_2/ReadVariableOp�
,batch_normalization_28/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_28_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_28/Cast_3/ReadVariableOp�
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_28/batchnorm/add/y�
$batch_normalization_28/batchnorm/addAddV24batch_normalization_28/Cast_1/ReadVariableOp:value:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_28/batchnorm/add�
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_28/batchnorm/Rsqrt�
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:04batch_normalization_28/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_28/batchnorm/mul�
&batch_normalization_28/batchnorm/mul_1Muldense_23/MatMul:product:0(batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_28/batchnorm/mul_1�
&batch_normalization_28/batchnorm/mul_2Mul2batch_normalization_28/Cast/ReadVariableOp:value:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_28/batchnorm/mul_2�
$batch_normalization_28/batchnorm/subSub4batch_normalization_28/Cast_2/ReadVariableOp:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_28/batchnorm/sub�
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_28/batchnorm/add_1v
Relu_3Relu*batch_normalization_28/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_24/MatMul/ReadVariableOp�
dense_24/MatMulMatMulRelu_3:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_24/MatMul�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_24/BiasAdd/ReadVariableOp�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_24/BiasAddt
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�	
NoOpNoOp+^batch_normalization_24/Cast/ReadVariableOp-^batch_normalization_24/Cast_1/ReadVariableOp-^batch_normalization_24/Cast_2/ReadVariableOp-^batch_normalization_24/Cast_3/ReadVariableOp+^batch_normalization_25/Cast/ReadVariableOp-^batch_normalization_25/Cast_1/ReadVariableOp-^batch_normalization_25/Cast_2/ReadVariableOp-^batch_normalization_25/Cast_3/ReadVariableOp+^batch_normalization_26/Cast/ReadVariableOp-^batch_normalization_26/Cast_1/ReadVariableOp-^batch_normalization_26/Cast_2/ReadVariableOp-^batch_normalization_26/Cast_3/ReadVariableOp+^batch_normalization_27/Cast/ReadVariableOp-^batch_normalization_27/Cast_1/ReadVariableOp-^batch_normalization_27/Cast_2/ReadVariableOp-^batch_normalization_27/Cast_3/ReadVariableOp+^batch_normalization_28/Cast/ReadVariableOp-^batch_normalization_28/Cast_1/ReadVariableOp-^batch_normalization_28/Cast_2/ReadVariableOp-^batch_normalization_28/Cast_3/ReadVariableOp^dense_20/MatMul/ReadVariableOp^dense_21/MatMul/ReadVariableOp^dense_22/MatMul/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_24/Cast/ReadVariableOp*batch_normalization_24/Cast/ReadVariableOp2\
,batch_normalization_24/Cast_1/ReadVariableOp,batch_normalization_24/Cast_1/ReadVariableOp2\
,batch_normalization_24/Cast_2/ReadVariableOp,batch_normalization_24/Cast_2/ReadVariableOp2\
,batch_normalization_24/Cast_3/ReadVariableOp,batch_normalization_24/Cast_3/ReadVariableOp2X
*batch_normalization_25/Cast/ReadVariableOp*batch_normalization_25/Cast/ReadVariableOp2\
,batch_normalization_25/Cast_1/ReadVariableOp,batch_normalization_25/Cast_1/ReadVariableOp2\
,batch_normalization_25/Cast_2/ReadVariableOp,batch_normalization_25/Cast_2/ReadVariableOp2\
,batch_normalization_25/Cast_3/ReadVariableOp,batch_normalization_25/Cast_3/ReadVariableOp2X
*batch_normalization_26/Cast/ReadVariableOp*batch_normalization_26/Cast/ReadVariableOp2\
,batch_normalization_26/Cast_1/ReadVariableOp,batch_normalization_26/Cast_1/ReadVariableOp2\
,batch_normalization_26/Cast_2/ReadVariableOp,batch_normalization_26/Cast_2/ReadVariableOp2\
,batch_normalization_26/Cast_3/ReadVariableOp,batch_normalization_26/Cast_3/ReadVariableOp2X
*batch_normalization_27/Cast/ReadVariableOp*batch_normalization_27/Cast/ReadVariableOp2\
,batch_normalization_27/Cast_1/ReadVariableOp,batch_normalization_27/Cast_1/ReadVariableOp2\
,batch_normalization_27/Cast_2/ReadVariableOp,batch_normalization_27/Cast_2/ReadVariableOp2\
,batch_normalization_27/Cast_3/ReadVariableOp,batch_normalization_27/Cast_3/ReadVariableOp2X
*batch_normalization_28/Cast/ReadVariableOp*batch_normalization_28/Cast/ReadVariableOp2\
,batch_normalization_28/Cast_1/ReadVariableOp,batch_normalization_28/Cast_1/ReadVariableOp2\
,batch_normalization_28/Cast_2/ReadVariableOp,batch_normalization_28/Cast_2/ReadVariableOp2\
,batch_normalization_28/Cast_3/ReadVariableOp,batch_normalization_28/Cast_3/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
�
�
7__inference_batch_normalization_26_layer_call_fn_317588

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
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3154152
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
7__inference_batch_normalization_24_layer_call_fn_317437

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
GPU 2J 8� *[
fVRT
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_3151452
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
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_317493

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
�
}
)__inference_dense_22_layer_call_fn_317807

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
GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_3159542
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
�+
�
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_317739

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
D__inference_dense_20_layer_call_and_return_conditional_losses_315912

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
�
�
7__inference_batch_normalization_27_layer_call_fn_317683

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
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_3156432
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
�
�
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_317703

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
�
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_317127
input_1L
>batch_normalization_24_assignmovingavg_readvariableop_resource:
N
@batch_normalization_24_assignmovingavg_1_readvariableop_resource:
A
3batch_normalization_24_cast_readvariableop_resource:
C
5batch_normalization_24_cast_1_readvariableop_resource:
9
'dense_20_matmul_readvariableop_resource:
L
>batch_normalization_25_assignmovingavg_readvariableop_resource:N
@batch_normalization_25_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_25_cast_readvariableop_resource:C
5batch_normalization_25_cast_1_readvariableop_resource:9
'dense_21_matmul_readvariableop_resource:L
>batch_normalization_26_assignmovingavg_readvariableop_resource:N
@batch_normalization_26_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_26_cast_readvariableop_resource:C
5batch_normalization_26_cast_1_readvariableop_resource:9
'dense_22_matmul_readvariableop_resource:L
>batch_normalization_27_assignmovingavg_readvariableop_resource:N
@batch_normalization_27_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_27_cast_readvariableop_resource:C
5batch_normalization_27_cast_1_readvariableop_resource:9
'dense_23_matmul_readvariableop_resource:L
>batch_normalization_28_assignmovingavg_readvariableop_resource:N
@batch_normalization_28_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_28_cast_readvariableop_resource:C
5batch_normalization_28_cast_1_readvariableop_resource:9
'dense_24_matmul_readvariableop_resource:
6
(dense_24_biasadd_readvariableop_resource:

identity��&batch_normalization_24/AssignMovingAvg�5batch_normalization_24/AssignMovingAvg/ReadVariableOp�(batch_normalization_24/AssignMovingAvg_1�7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_24/Cast/ReadVariableOp�,batch_normalization_24/Cast_1/ReadVariableOp�&batch_normalization_25/AssignMovingAvg�5batch_normalization_25/AssignMovingAvg/ReadVariableOp�(batch_normalization_25/AssignMovingAvg_1�7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_25/Cast/ReadVariableOp�,batch_normalization_25/Cast_1/ReadVariableOp�&batch_normalization_26/AssignMovingAvg�5batch_normalization_26/AssignMovingAvg/ReadVariableOp�(batch_normalization_26/AssignMovingAvg_1�7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_26/Cast/ReadVariableOp�,batch_normalization_26/Cast_1/ReadVariableOp�&batch_normalization_27/AssignMovingAvg�5batch_normalization_27/AssignMovingAvg/ReadVariableOp�(batch_normalization_27/AssignMovingAvg_1�7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_27/Cast/ReadVariableOp�,batch_normalization_27/Cast_1/ReadVariableOp�&batch_normalization_28/AssignMovingAvg�5batch_normalization_28/AssignMovingAvg/ReadVariableOp�(batch_normalization_28/AssignMovingAvg_1�7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_28/Cast/ReadVariableOp�,batch_normalization_28/Cast_1/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�
5batch_normalization_24/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_24/moments/mean/reduction_indices�
#batch_normalization_24/moments/meanMeaninput_1>batch_normalization_24/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_24/moments/mean�
+batch_normalization_24/moments/StopGradientStopGradient,batch_normalization_24/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_24/moments/StopGradient�
0batch_normalization_24/moments/SquaredDifferenceSquaredDifferenceinput_14batch_normalization_24/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������
22
0batch_normalization_24/moments/SquaredDifference�
9batch_normalization_24/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_24/moments/variance/reduction_indices�
'batch_normalization_24/moments/varianceMean4batch_normalization_24/moments/SquaredDifference:z:0Bbatch_normalization_24/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_24/moments/variance�
&batch_normalization_24/moments/SqueezeSqueeze,batch_normalization_24/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_24/moments/Squeeze�
(batch_normalization_24/moments/Squeeze_1Squeeze0batch_normalization_24/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_24/moments/Squeeze_1�
,batch_normalization_24/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_24/AssignMovingAvg/decay�
+batch_normalization_24/AssignMovingAvg/CastCast5batch_normalization_24/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_24/AssignMovingAvg/Cast�
5batch_normalization_24/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_24_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_24/AssignMovingAvg/ReadVariableOp�
*batch_normalization_24/AssignMovingAvg/subSub=batch_normalization_24/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_24/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_24/AssignMovingAvg/sub�
*batch_normalization_24/AssignMovingAvg/mulMul.batch_normalization_24/AssignMovingAvg/sub:z:0/batch_normalization_24/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_24/AssignMovingAvg/mul�
&batch_normalization_24/AssignMovingAvgAssignSubVariableOp>batch_normalization_24_assignmovingavg_readvariableop_resource.batch_normalization_24/AssignMovingAvg/mul:z:06^batch_normalization_24/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_24/AssignMovingAvg�
.batch_normalization_24/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_24/AssignMovingAvg_1/decay�
-batch_normalization_24/AssignMovingAvg_1/CastCast7batch_normalization_24/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_24/AssignMovingAvg_1/Cast�
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_24_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_24/AssignMovingAvg_1/subSub?batch_normalization_24/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_24/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_24/AssignMovingAvg_1/sub�
,batch_normalization_24/AssignMovingAvg_1/mulMul0batch_normalization_24/AssignMovingAvg_1/sub:z:01batch_normalization_24/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_24/AssignMovingAvg_1/mul�
(batch_normalization_24/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_24_assignmovingavg_1_readvariableop_resource0batch_normalization_24/AssignMovingAvg_1/mul:z:08^batch_normalization_24/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_24/AssignMovingAvg_1�
*batch_normalization_24/Cast/ReadVariableOpReadVariableOp3batch_normalization_24_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_24/Cast/ReadVariableOp�
,batch_normalization_24/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_24_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_24/Cast_1/ReadVariableOp�
&batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_24/batchnorm/add/y�
$batch_normalization_24/batchnorm/addAddV21batch_normalization_24/moments/Squeeze_1:output:0/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_24/batchnorm/add�
&batch_normalization_24/batchnorm/RsqrtRsqrt(batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_24/batchnorm/Rsqrt�
$batch_normalization_24/batchnorm/mulMul*batch_normalization_24/batchnorm/Rsqrt:y:04batch_normalization_24/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_24/batchnorm/mul�
&batch_normalization_24/batchnorm/mul_1Mulinput_1(batch_normalization_24/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_24/batchnorm/mul_1�
&batch_normalization_24/batchnorm/mul_2Mul/batch_normalization_24/moments/Squeeze:output:0(batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_24/batchnorm/mul_2�
$batch_normalization_24/batchnorm/subSub2batch_normalization_24/Cast/ReadVariableOp:value:0*batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_24/batchnorm/sub�
&batch_normalization_24/batchnorm/add_1AddV2*batch_normalization_24/batchnorm/mul_1:z:0(batch_normalization_24/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_24/batchnorm/add_1�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMul*batch_normalization_24/batchnorm/add_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
5batch_normalization_25/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_25/moments/mean/reduction_indices�
#batch_normalization_25/moments/meanMeandense_20/MatMul:product:0>batch_normalization_25/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_25/moments/mean�
+batch_normalization_25/moments/StopGradientStopGradient,batch_normalization_25/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_25/moments/StopGradient�
0batch_normalization_25/moments/SquaredDifferenceSquaredDifferencedense_20/MatMul:product:04batch_normalization_25/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_25/moments/SquaredDifference�
9batch_normalization_25/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_25/moments/variance/reduction_indices�
'batch_normalization_25/moments/varianceMean4batch_normalization_25/moments/SquaredDifference:z:0Bbatch_normalization_25/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_25/moments/variance�
&batch_normalization_25/moments/SqueezeSqueeze,batch_normalization_25/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_25/moments/Squeeze�
(batch_normalization_25/moments/Squeeze_1Squeeze0batch_normalization_25/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_25/moments/Squeeze_1�
,batch_normalization_25/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_25/AssignMovingAvg/decay�
+batch_normalization_25/AssignMovingAvg/CastCast5batch_normalization_25/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_25/AssignMovingAvg/Cast�
5batch_normalization_25/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_25_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_25/AssignMovingAvg/ReadVariableOp�
*batch_normalization_25/AssignMovingAvg/subSub=batch_normalization_25/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_25/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_25/AssignMovingAvg/sub�
*batch_normalization_25/AssignMovingAvg/mulMul.batch_normalization_25/AssignMovingAvg/sub:z:0/batch_normalization_25/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_25/AssignMovingAvg/mul�
&batch_normalization_25/AssignMovingAvgAssignSubVariableOp>batch_normalization_25_assignmovingavg_readvariableop_resource.batch_normalization_25/AssignMovingAvg/mul:z:06^batch_normalization_25/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_25/AssignMovingAvg�
.batch_normalization_25/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_25/AssignMovingAvg_1/decay�
-batch_normalization_25/AssignMovingAvg_1/CastCast7batch_normalization_25/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_25/AssignMovingAvg_1/Cast�
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_25_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_25/AssignMovingAvg_1/subSub?batch_normalization_25/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_25/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_25/AssignMovingAvg_1/sub�
,batch_normalization_25/AssignMovingAvg_1/mulMul0batch_normalization_25/AssignMovingAvg_1/sub:z:01batch_normalization_25/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_25/AssignMovingAvg_1/mul�
(batch_normalization_25/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_25_assignmovingavg_1_readvariableop_resource0batch_normalization_25/AssignMovingAvg_1/mul:z:08^batch_normalization_25/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_25/AssignMovingAvg_1�
*batch_normalization_25/Cast/ReadVariableOpReadVariableOp3batch_normalization_25_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_25/Cast/ReadVariableOp�
,batch_normalization_25/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_25_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_25/Cast_1/ReadVariableOp�
&batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_25/batchnorm/add/y�
$batch_normalization_25/batchnorm/addAddV21batch_normalization_25/moments/Squeeze_1:output:0/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_25/batchnorm/add�
&batch_normalization_25/batchnorm/RsqrtRsqrt(batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_25/batchnorm/Rsqrt�
$batch_normalization_25/batchnorm/mulMul*batch_normalization_25/batchnorm/Rsqrt:y:04batch_normalization_25/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_25/batchnorm/mul�
&batch_normalization_25/batchnorm/mul_1Muldense_20/MatMul:product:0(batch_normalization_25/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_25/batchnorm/mul_1�
&batch_normalization_25/batchnorm/mul_2Mul/batch_normalization_25/moments/Squeeze:output:0(batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_25/batchnorm/mul_2�
$batch_normalization_25/batchnorm/subSub2batch_normalization_25/Cast/ReadVariableOp:value:0*batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_25/batchnorm/sub�
&batch_normalization_25/batchnorm/add_1AddV2*batch_normalization_25/batchnorm/mul_1:z:0(batch_normalization_25/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_25/batchnorm/add_1r
ReluRelu*batch_normalization_25/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_21/MatMul/ReadVariableOp�
dense_21/MatMulMatMulRelu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_21/MatMul�
5batch_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_26/moments/mean/reduction_indices�
#batch_normalization_26/moments/meanMeandense_21/MatMul:product:0>batch_normalization_26/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_26/moments/mean�
+batch_normalization_26/moments/StopGradientStopGradient,batch_normalization_26/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_26/moments/StopGradient�
0batch_normalization_26/moments/SquaredDifferenceSquaredDifferencedense_21/MatMul:product:04batch_normalization_26/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_26/moments/SquaredDifference�
9batch_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_26/moments/variance/reduction_indices�
'batch_normalization_26/moments/varianceMean4batch_normalization_26/moments/SquaredDifference:z:0Bbatch_normalization_26/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_26/moments/variance�
&batch_normalization_26/moments/SqueezeSqueeze,batch_normalization_26/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_26/moments/Squeeze�
(batch_normalization_26/moments/Squeeze_1Squeeze0batch_normalization_26/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_26/moments/Squeeze_1�
,batch_normalization_26/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_26/AssignMovingAvg/decay�
+batch_normalization_26/AssignMovingAvg/CastCast5batch_normalization_26/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_26/AssignMovingAvg/Cast�
5batch_normalization_26/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_26_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_26/AssignMovingAvg/ReadVariableOp�
*batch_normalization_26/AssignMovingAvg/subSub=batch_normalization_26/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_26/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_26/AssignMovingAvg/sub�
*batch_normalization_26/AssignMovingAvg/mulMul.batch_normalization_26/AssignMovingAvg/sub:z:0/batch_normalization_26/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_26/AssignMovingAvg/mul�
&batch_normalization_26/AssignMovingAvgAssignSubVariableOp>batch_normalization_26_assignmovingavg_readvariableop_resource.batch_normalization_26/AssignMovingAvg/mul:z:06^batch_normalization_26/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_26/AssignMovingAvg�
.batch_normalization_26/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_26/AssignMovingAvg_1/decay�
-batch_normalization_26/AssignMovingAvg_1/CastCast7batch_normalization_26/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_26/AssignMovingAvg_1/Cast�
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_26_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_26/AssignMovingAvg_1/subSub?batch_normalization_26/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_26/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_26/AssignMovingAvg_1/sub�
,batch_normalization_26/AssignMovingAvg_1/mulMul0batch_normalization_26/AssignMovingAvg_1/sub:z:01batch_normalization_26/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_26/AssignMovingAvg_1/mul�
(batch_normalization_26/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_26_assignmovingavg_1_readvariableop_resource0batch_normalization_26/AssignMovingAvg_1/mul:z:08^batch_normalization_26/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_26/AssignMovingAvg_1�
*batch_normalization_26/Cast/ReadVariableOpReadVariableOp3batch_normalization_26_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_26/Cast/ReadVariableOp�
,batch_normalization_26/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_26_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_26/Cast_1/ReadVariableOp�
&batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_26/batchnorm/add/y�
$batch_normalization_26/batchnorm/addAddV21batch_normalization_26/moments/Squeeze_1:output:0/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_26/batchnorm/add�
&batch_normalization_26/batchnorm/RsqrtRsqrt(batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_26/batchnorm/Rsqrt�
$batch_normalization_26/batchnorm/mulMul*batch_normalization_26/batchnorm/Rsqrt:y:04batch_normalization_26/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_26/batchnorm/mul�
&batch_normalization_26/batchnorm/mul_1Muldense_21/MatMul:product:0(batch_normalization_26/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_26/batchnorm/mul_1�
&batch_normalization_26/batchnorm/mul_2Mul/batch_normalization_26/moments/Squeeze:output:0(batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_26/batchnorm/mul_2�
$batch_normalization_26/batchnorm/subSub2batch_normalization_26/Cast/ReadVariableOp:value:0*batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_26/batchnorm/sub�
&batch_normalization_26/batchnorm/add_1AddV2*batch_normalization_26/batchnorm/mul_1:z:0(batch_normalization_26/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_26/batchnorm/add_1v
Relu_1Relu*batch_normalization_26/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_22/MatMul/ReadVariableOp�
dense_22/MatMulMatMulRelu_1:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_22/MatMul�
5batch_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_27/moments/mean/reduction_indices�
#batch_normalization_27/moments/meanMeandense_22/MatMul:product:0>batch_normalization_27/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_27/moments/mean�
+batch_normalization_27/moments/StopGradientStopGradient,batch_normalization_27/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_27/moments/StopGradient�
0batch_normalization_27/moments/SquaredDifferenceSquaredDifferencedense_22/MatMul:product:04batch_normalization_27/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_27/moments/SquaredDifference�
9batch_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_27/moments/variance/reduction_indices�
'batch_normalization_27/moments/varianceMean4batch_normalization_27/moments/SquaredDifference:z:0Bbatch_normalization_27/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_27/moments/variance�
&batch_normalization_27/moments/SqueezeSqueeze,batch_normalization_27/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_27/moments/Squeeze�
(batch_normalization_27/moments/Squeeze_1Squeeze0batch_normalization_27/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_27/moments/Squeeze_1�
,batch_normalization_27/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_27/AssignMovingAvg/decay�
+batch_normalization_27/AssignMovingAvg/CastCast5batch_normalization_27/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_27/AssignMovingAvg/Cast�
5batch_normalization_27/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_27_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_27/AssignMovingAvg/ReadVariableOp�
*batch_normalization_27/AssignMovingAvg/subSub=batch_normalization_27/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_27/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_27/AssignMovingAvg/sub�
*batch_normalization_27/AssignMovingAvg/mulMul.batch_normalization_27/AssignMovingAvg/sub:z:0/batch_normalization_27/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_27/AssignMovingAvg/mul�
&batch_normalization_27/AssignMovingAvgAssignSubVariableOp>batch_normalization_27_assignmovingavg_readvariableop_resource.batch_normalization_27/AssignMovingAvg/mul:z:06^batch_normalization_27/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_27/AssignMovingAvg�
.batch_normalization_27/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_27/AssignMovingAvg_1/decay�
-batch_normalization_27/AssignMovingAvg_1/CastCast7batch_normalization_27/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_27/AssignMovingAvg_1/Cast�
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_27_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_27/AssignMovingAvg_1/subSub?batch_normalization_27/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_27/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_27/AssignMovingAvg_1/sub�
,batch_normalization_27/AssignMovingAvg_1/mulMul0batch_normalization_27/AssignMovingAvg_1/sub:z:01batch_normalization_27/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_27/AssignMovingAvg_1/mul�
(batch_normalization_27/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_27_assignmovingavg_1_readvariableop_resource0batch_normalization_27/AssignMovingAvg_1/mul:z:08^batch_normalization_27/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_27/AssignMovingAvg_1�
*batch_normalization_27/Cast/ReadVariableOpReadVariableOp3batch_normalization_27_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_27/Cast/ReadVariableOp�
,batch_normalization_27/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_27_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_27/Cast_1/ReadVariableOp�
&batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_27/batchnorm/add/y�
$batch_normalization_27/batchnorm/addAddV21batch_normalization_27/moments/Squeeze_1:output:0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_27/batchnorm/add�
&batch_normalization_27/batchnorm/RsqrtRsqrt(batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_27/batchnorm/Rsqrt�
$batch_normalization_27/batchnorm/mulMul*batch_normalization_27/batchnorm/Rsqrt:y:04batch_normalization_27/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_27/batchnorm/mul�
&batch_normalization_27/batchnorm/mul_1Muldense_22/MatMul:product:0(batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_27/batchnorm/mul_1�
&batch_normalization_27/batchnorm/mul_2Mul/batch_normalization_27/moments/Squeeze:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_27/batchnorm/mul_2�
$batch_normalization_27/batchnorm/subSub2batch_normalization_27/Cast/ReadVariableOp:value:0*batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_27/batchnorm/sub�
&batch_normalization_27/batchnorm/add_1AddV2*batch_normalization_27/batchnorm/mul_1:z:0(batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_27/batchnorm/add_1v
Relu_2Relu*batch_normalization_27/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_23/MatMul/ReadVariableOp�
dense_23/MatMulMatMulRelu_2:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_23/MatMul�
5batch_normalization_28/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_28/moments/mean/reduction_indices�
#batch_normalization_28/moments/meanMeandense_23/MatMul:product:0>batch_normalization_28/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_28/moments/mean�
+batch_normalization_28/moments/StopGradientStopGradient,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_28/moments/StopGradient�
0batch_normalization_28/moments/SquaredDifferenceSquaredDifferencedense_23/MatMul:product:04batch_normalization_28/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_28/moments/SquaredDifference�
9batch_normalization_28/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_28/moments/variance/reduction_indices�
'batch_normalization_28/moments/varianceMean4batch_normalization_28/moments/SquaredDifference:z:0Bbatch_normalization_28/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_28/moments/variance�
&batch_normalization_28/moments/SqueezeSqueeze,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_28/moments/Squeeze�
(batch_normalization_28/moments/Squeeze_1Squeeze0batch_normalization_28/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_28/moments/Squeeze_1�
,batch_normalization_28/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_28/AssignMovingAvg/decay�
+batch_normalization_28/AssignMovingAvg/CastCast5batch_normalization_28/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_28/AssignMovingAvg/Cast�
5batch_normalization_28/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_28_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_28/AssignMovingAvg/ReadVariableOp�
*batch_normalization_28/AssignMovingAvg/subSub=batch_normalization_28/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_28/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_28/AssignMovingAvg/sub�
*batch_normalization_28/AssignMovingAvg/mulMul.batch_normalization_28/AssignMovingAvg/sub:z:0/batch_normalization_28/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_28/AssignMovingAvg/mul�
&batch_normalization_28/AssignMovingAvgAssignSubVariableOp>batch_normalization_28_assignmovingavg_readvariableop_resource.batch_normalization_28/AssignMovingAvg/mul:z:06^batch_normalization_28/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_28/AssignMovingAvg�
.batch_normalization_28/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_28/AssignMovingAvg_1/decay�
-batch_normalization_28/AssignMovingAvg_1/CastCast7batch_normalization_28/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_28/AssignMovingAvg_1/Cast�
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_28/AssignMovingAvg_1/subSub?batch_normalization_28/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_28/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_28/AssignMovingAvg_1/sub�
,batch_normalization_28/AssignMovingAvg_1/mulMul0batch_normalization_28/AssignMovingAvg_1/sub:z:01batch_normalization_28/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_28/AssignMovingAvg_1/mul�
(batch_normalization_28/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource0batch_normalization_28/AssignMovingAvg_1/mul:z:08^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_28/AssignMovingAvg_1�
*batch_normalization_28/Cast/ReadVariableOpReadVariableOp3batch_normalization_28_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_28/Cast/ReadVariableOp�
,batch_normalization_28/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_28_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_28/Cast_1/ReadVariableOp�
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_28/batchnorm/add/y�
$batch_normalization_28/batchnorm/addAddV21batch_normalization_28/moments/Squeeze_1:output:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_28/batchnorm/add�
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_28/batchnorm/Rsqrt�
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:04batch_normalization_28/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_28/batchnorm/mul�
&batch_normalization_28/batchnorm/mul_1Muldense_23/MatMul:product:0(batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_28/batchnorm/mul_1�
&batch_normalization_28/batchnorm/mul_2Mul/batch_normalization_28/moments/Squeeze:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_28/batchnorm/mul_2�
$batch_normalization_28/batchnorm/subSub2batch_normalization_28/Cast/ReadVariableOp:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_28/batchnorm/sub�
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_28/batchnorm/add_1v
Relu_3Relu*batch_normalization_28/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_24/MatMul/ReadVariableOp�
dense_24/MatMulMatMulRelu_3:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_24/MatMul�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_24/BiasAdd/ReadVariableOp�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_24/BiasAddt
IdentityIdentitydense_24/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp'^batch_normalization_24/AssignMovingAvg6^batch_normalization_24/AssignMovingAvg/ReadVariableOp)^batch_normalization_24/AssignMovingAvg_18^batch_normalization_24/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_24/Cast/ReadVariableOp-^batch_normalization_24/Cast_1/ReadVariableOp'^batch_normalization_25/AssignMovingAvg6^batch_normalization_25/AssignMovingAvg/ReadVariableOp)^batch_normalization_25/AssignMovingAvg_18^batch_normalization_25/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_25/Cast/ReadVariableOp-^batch_normalization_25/Cast_1/ReadVariableOp'^batch_normalization_26/AssignMovingAvg6^batch_normalization_26/AssignMovingAvg/ReadVariableOp)^batch_normalization_26/AssignMovingAvg_18^batch_normalization_26/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_26/Cast/ReadVariableOp-^batch_normalization_26/Cast_1/ReadVariableOp'^batch_normalization_27/AssignMovingAvg6^batch_normalization_27/AssignMovingAvg/ReadVariableOp)^batch_normalization_27/AssignMovingAvg_18^batch_normalization_27/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_27/Cast/ReadVariableOp-^batch_normalization_27/Cast_1/ReadVariableOp'^batch_normalization_28/AssignMovingAvg6^batch_normalization_28/AssignMovingAvg/ReadVariableOp)^batch_normalization_28/AssignMovingAvg_18^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_28/Cast/ReadVariableOp-^batch_normalization_28/Cast_1/ReadVariableOp^dense_20/MatMul/ReadVariableOp^dense_21/MatMul/ReadVariableOp^dense_22/MatMul/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_24/AssignMovingAvg&batch_normalization_24/AssignMovingAvg2n
5batch_normalization_24/AssignMovingAvg/ReadVariableOp5batch_normalization_24/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_24/AssignMovingAvg_1(batch_normalization_24/AssignMovingAvg_12r
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_24/Cast/ReadVariableOp*batch_normalization_24/Cast/ReadVariableOp2\
,batch_normalization_24/Cast_1/ReadVariableOp,batch_normalization_24/Cast_1/ReadVariableOp2P
&batch_normalization_25/AssignMovingAvg&batch_normalization_25/AssignMovingAvg2n
5batch_normalization_25/AssignMovingAvg/ReadVariableOp5batch_normalization_25/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_25/AssignMovingAvg_1(batch_normalization_25/AssignMovingAvg_12r
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_25/Cast/ReadVariableOp*batch_normalization_25/Cast/ReadVariableOp2\
,batch_normalization_25/Cast_1/ReadVariableOp,batch_normalization_25/Cast_1/ReadVariableOp2P
&batch_normalization_26/AssignMovingAvg&batch_normalization_26/AssignMovingAvg2n
5batch_normalization_26/AssignMovingAvg/ReadVariableOp5batch_normalization_26/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_26/AssignMovingAvg_1(batch_normalization_26/AssignMovingAvg_12r
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_26/Cast/ReadVariableOp*batch_normalization_26/Cast/ReadVariableOp2\
,batch_normalization_26/Cast_1/ReadVariableOp,batch_normalization_26/Cast_1/ReadVariableOp2P
&batch_normalization_27/AssignMovingAvg&batch_normalization_27/AssignMovingAvg2n
5batch_normalization_27/AssignMovingAvg/ReadVariableOp5batch_normalization_27/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_27/AssignMovingAvg_1(batch_normalization_27/AssignMovingAvg_12r
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_27/Cast/ReadVariableOp*batch_normalization_27/Cast/ReadVariableOp2\
,batch_normalization_27/Cast_1/ReadVariableOp,batch_normalization_27/Cast_1/ReadVariableOp2P
&batch_normalization_28/AssignMovingAvg&batch_normalization_28/AssignMovingAvg2n
5batch_normalization_28/AssignMovingAvg/ReadVariableOp5batch_normalization_28/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_28/AssignMovingAvg_1(batch_normalization_28/AssignMovingAvg_12r
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_28/Cast/ReadVariableOp*batch_normalization_28/Cast/ReadVariableOp2\
,batch_normalization_28/Cast_1/ReadVariableOp,batch_normalization_28/Cast_1/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
�
�
7__inference_feed_forward_sub_net_4_layer_call_fn_317184
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
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_3160062
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
�
�
7__inference_batch_normalization_25_layer_call_fn_317506

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
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3152492
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
D__inference_dense_22_layer_call_and_return_conditional_losses_317800

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
7__inference_batch_normalization_25_layer_call_fn_317519

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
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_3153112
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
}
)__inference_dense_23_layer_call_fn_317821

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
GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_3159752
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
D__inference_dense_24_layer_call_and_return_conditional_losses_317831

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
�z
�
"__inference__traced_restore_318029
file_prefixR
Dassignvariableop_feed_forward_sub_net_4_batch_normalization_24_gamma:
S
Eassignvariableop_1_feed_forward_sub_net_4_batch_normalization_24_beta:
T
Fassignvariableop_2_feed_forward_sub_net_4_batch_normalization_25_gamma:S
Eassignvariableop_3_feed_forward_sub_net_4_batch_normalization_25_beta:T
Fassignvariableop_4_feed_forward_sub_net_4_batch_normalization_26_gamma:S
Eassignvariableop_5_feed_forward_sub_net_4_batch_normalization_26_beta:T
Fassignvariableop_6_feed_forward_sub_net_4_batch_normalization_27_gamma:S
Eassignvariableop_7_feed_forward_sub_net_4_batch_normalization_27_beta:T
Fassignvariableop_8_feed_forward_sub_net_4_batch_normalization_28_gamma:S
Eassignvariableop_9_feed_forward_sub_net_4_batch_normalization_28_beta:[
Massignvariableop_10_feed_forward_sub_net_4_batch_normalization_24_moving_mean:
_
Qassignvariableop_11_feed_forward_sub_net_4_batch_normalization_24_moving_variance:
[
Massignvariableop_12_feed_forward_sub_net_4_batch_normalization_25_moving_mean:_
Qassignvariableop_13_feed_forward_sub_net_4_batch_normalization_25_moving_variance:[
Massignvariableop_14_feed_forward_sub_net_4_batch_normalization_26_moving_mean:_
Qassignvariableop_15_feed_forward_sub_net_4_batch_normalization_26_moving_variance:[
Massignvariableop_16_feed_forward_sub_net_4_batch_normalization_27_moving_mean:_
Qassignvariableop_17_feed_forward_sub_net_4_batch_normalization_27_moving_variance:[
Massignvariableop_18_feed_forward_sub_net_4_batch_normalization_28_moving_mean:_
Qassignvariableop_19_feed_forward_sub_net_4_batch_normalization_28_moving_variance:L
:assignvariableop_20_feed_forward_sub_net_4_dense_20_kernel:
L
:assignvariableop_21_feed_forward_sub_net_4_dense_21_kernel:L
:assignvariableop_22_feed_forward_sub_net_4_dense_22_kernel:L
:assignvariableop_23_feed_forward_sub_net_4_dense_23_kernel:L
:assignvariableop_24_feed_forward_sub_net_4_dense_24_kernel:
F
8assignvariableop_25_feed_forward_sub_net_4_dense_24_bias:
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
AssignVariableOpAssignVariableOpDassignvariableop_feed_forward_sub_net_4_batch_normalization_24_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpEassignvariableop_1_feed_forward_sub_net_4_batch_normalization_24_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpFassignvariableop_2_feed_forward_sub_net_4_batch_normalization_25_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpEassignvariableop_3_feed_forward_sub_net_4_batch_normalization_25_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpFassignvariableop_4_feed_forward_sub_net_4_batch_normalization_26_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpEassignvariableop_5_feed_forward_sub_net_4_batch_normalization_26_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpFassignvariableop_6_feed_forward_sub_net_4_batch_normalization_27_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpEassignvariableop_7_feed_forward_sub_net_4_batch_normalization_27_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpFassignvariableop_8_feed_forward_sub_net_4_batch_normalization_28_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpEassignvariableop_9_feed_forward_sub_net_4_batch_normalization_28_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpMassignvariableop_10_feed_forward_sub_net_4_batch_normalization_24_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpQassignvariableop_11_feed_forward_sub_net_4_batch_normalization_24_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpMassignvariableop_12_feed_forward_sub_net_4_batch_normalization_25_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpQassignvariableop_13_feed_forward_sub_net_4_batch_normalization_25_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpMassignvariableop_14_feed_forward_sub_net_4_batch_normalization_26_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpQassignvariableop_15_feed_forward_sub_net_4_batch_normalization_26_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpMassignvariableop_16_feed_forward_sub_net_4_batch_normalization_27_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpQassignvariableop_17_feed_forward_sub_net_4_batch_normalization_27_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpMassignvariableop_18_feed_forward_sub_net_4_batch_normalization_28_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpQassignvariableop_19_feed_forward_sub_net_4_batch_normalization_28_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp:assignvariableop_20_feed_forward_sub_net_4_dense_20_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp:assignvariableop_21_feed_forward_sub_net_4_dense_21_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp:assignvariableop_22_feed_forward_sub_net_4_dense_22_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp:assignvariableop_23_feed_forward_sub_net_4_dense_23_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp:assignvariableop_24_feed_forward_sub_net_4_dense_24_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp8assignvariableop_25_feed_forward_sub_net_4_dense_24_biasIdentity_25:output:0"/device:CPU:0*
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
�
�
7__inference_batch_normalization_28_layer_call_fn_317752

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
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_3157472
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
}
)__inference_dense_20_layer_call_fn_317779

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
GPU 2J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_3159122
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
�
�
)__inference_dense_24_layer_call_fn_317840

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
GPU 2J 8� *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_3159992
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
�
�
D__inference_dense_21_layer_call_and_return_conditional_losses_315933

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
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_315415

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
7__inference_batch_normalization_26_layer_call_fn_317601

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
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_3154772
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
7__inference_batch_normalization_28_layer_call_fn_317765

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
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_3158092
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
A:?
23feed_forward_sub_net_4/batch_normalization_24/gamma
@:>
22feed_forward_sub_net_4/batch_normalization_24/beta
A:?23feed_forward_sub_net_4/batch_normalization_25/gamma
@:>22feed_forward_sub_net_4/batch_normalization_25/beta
A:?23feed_forward_sub_net_4/batch_normalization_26/gamma
@:>22feed_forward_sub_net_4/batch_normalization_26/beta
A:?23feed_forward_sub_net_4/batch_normalization_27/gamma
@:>22feed_forward_sub_net_4/batch_normalization_27/beta
A:?23feed_forward_sub_net_4/batch_normalization_28/gamma
@:>22feed_forward_sub_net_4/batch_normalization_28/beta
I:G
 (29feed_forward_sub_net_4/batch_normalization_24/moving_mean
M:K
 (2=feed_forward_sub_net_4/batch_normalization_24/moving_variance
I:G (29feed_forward_sub_net_4/batch_normalization_25/moving_mean
M:K (2=feed_forward_sub_net_4/batch_normalization_25/moving_variance
I:G (29feed_forward_sub_net_4/batch_normalization_26/moving_mean
M:K (2=feed_forward_sub_net_4/batch_normalization_26/moving_variance
I:G (29feed_forward_sub_net_4/batch_normalization_27/moving_mean
M:K (2=feed_forward_sub_net_4/batch_normalization_27/moving_variance
I:G (29feed_forward_sub_net_4/batch_normalization_28/moving_mean
M:K (2=feed_forward_sub_net_4/batch_normalization_28/moving_variance
8:6
2&feed_forward_sub_net_4/dense_20/kernel
8:62&feed_forward_sub_net_4/dense_21/kernel
8:62&feed_forward_sub_net_4/dense_22/kernel
8:62&feed_forward_sub_net_4/dense_23/kernel
8:6
2&feed_forward_sub_net_4/dense_24/kernel
2:0
2$feed_forward_sub_net_4/dense_24/bias
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
!__inference__wrapped_model_315059input_1"�
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
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_316649
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_316835
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_316941
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_317127�
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
7__inference_feed_forward_sub_net_4_layer_call_fn_317184
7__inference_feed_forward_sub_net_4_layer_call_fn_317241
7__inference_feed_forward_sub_net_4_layer_call_fn_317298
7__inference_feed_forward_sub_net_4_layer_call_fn_317355�
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
$__inference_signature_wrapper_316543input_1"�
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
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_317375
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_317411�
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
7__inference_batch_normalization_24_layer_call_fn_317424
7__inference_batch_normalization_24_layer_call_fn_317437�
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
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_317457
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_317493�
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
7__inference_batch_normalization_25_layer_call_fn_317506
7__inference_batch_normalization_25_layer_call_fn_317519�
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
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_317539
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_317575�
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
7__inference_batch_normalization_26_layer_call_fn_317588
7__inference_batch_normalization_26_layer_call_fn_317601�
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
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_317621
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_317657�
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
7__inference_batch_normalization_27_layer_call_fn_317670
7__inference_batch_normalization_27_layer_call_fn_317683�
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
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_317703
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_317739�
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
7__inference_batch_normalization_28_layer_call_fn_317752
7__inference_batch_normalization_28_layer_call_fn_317765�
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
D__inference_dense_20_layer_call_and_return_conditional_losses_317772�
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
)__inference_dense_20_layer_call_fn_317779�
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
D__inference_dense_21_layer_call_and_return_conditional_losses_317786�
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
)__inference_dense_21_layer_call_fn_317793�
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
D__inference_dense_22_layer_call_and_return_conditional_losses_317800�
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
)__inference_dense_22_layer_call_fn_317807�
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
D__inference_dense_23_layer_call_and_return_conditional_losses_317814�
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
)__inference_dense_23_layer_call_fn_317821�
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
D__inference_dense_24_layer_call_and_return_conditional_losses_317831�
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
)__inference_dense_24_layer_call_fn_317840�
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
!__inference__wrapped_model_315059�' (!")#$*%&+,0�-
&�#
!�
input_1���������

� "3�0
.
output_1"�
output_1���������
�
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_317375b3�0
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
R__inference_batch_normalization_24_layer_call_and_return_conditional_losses_317411b3�0
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
7__inference_batch_normalization_24_layer_call_fn_317424U3�0
)�&
 �
inputs���������

p 
� "����������
�
7__inference_batch_normalization_24_layer_call_fn_317437U3�0
)�&
 �
inputs���������

p
� "����������
�
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_317457b 3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_25_layer_call_and_return_conditional_losses_317493b 3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_25_layer_call_fn_317506U 3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_25_layer_call_fn_317519U 3�0
)�&
 �
inputs���������
p
� "�����������
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_317539b!"3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_26_layer_call_and_return_conditional_losses_317575b!"3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_26_layer_call_fn_317588U!"3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_26_layer_call_fn_317601U!"3�0
)�&
 �
inputs���������
p
� "�����������
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_317621b#$3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_27_layer_call_and_return_conditional_losses_317657b#$3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_27_layer_call_fn_317670U#$3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_27_layer_call_fn_317683U#$3�0
)�&
 �
inputs���������
p
� "�����������
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_317703b%&3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_28_layer_call_and_return_conditional_losses_317739b%&3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_28_layer_call_fn_317752U%&3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_28_layer_call_fn_317765U%&3�0
)�&
 �
inputs���������
p
� "�����������
D__inference_dense_20_layer_call_and_return_conditional_losses_317772['/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� {
)__inference_dense_20_layer_call_fn_317779N'/�,
%�"
 �
inputs���������

� "�����������
D__inference_dense_21_layer_call_and_return_conditional_losses_317786[(/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
)__inference_dense_21_layer_call_fn_317793N(/�,
%�"
 �
inputs���������
� "�����������
D__inference_dense_22_layer_call_and_return_conditional_losses_317800[)/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
)__inference_dense_22_layer_call_fn_317807N)/�,
%�"
 �
inputs���������
� "�����������
D__inference_dense_23_layer_call_and_return_conditional_losses_317814[*/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
)__inference_dense_23_layer_call_fn_317821N*/�,
%�"
 �
inputs���������
� "�����������
D__inference_dense_24_layer_call_and_return_conditional_losses_317831\+,/�,
%�"
 �
inputs���������
� "%�"
�
0���������

� |
)__inference_dense_24_layer_call_fn_317840O+,/�,
%�"
 �
inputs���������
� "����������
�
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_316649s' (!")#$*%&+,.�+
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
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_316835s' (!")#$*%&+,.�+
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
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_316941y' (!")#$*%&+,4�1
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
R__inference_feed_forward_sub_net_4_layer_call_and_return_conditional_losses_317127y' (!")#$*%&+,4�1
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
7__inference_feed_forward_sub_net_4_layer_call_fn_317184l' (!")#$*%&+,4�1
*�'
!�
input_1���������

p 
� "����������
�
7__inference_feed_forward_sub_net_4_layer_call_fn_317241f' (!")#$*%&+,.�+
$�!
�
x���������

p 
� "����������
�
7__inference_feed_forward_sub_net_4_layer_call_fn_317298f' (!")#$*%&+,.�+
$�!
�
x���������

p
� "����������
�
7__inference_feed_forward_sub_net_4_layer_call_fn_317355l' (!")#$*%&+,4�1
*�'
!�
input_1���������

p
� "����������
�
$__inference_signature_wrapper_316543�' (!")#$*%&+,;�8
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