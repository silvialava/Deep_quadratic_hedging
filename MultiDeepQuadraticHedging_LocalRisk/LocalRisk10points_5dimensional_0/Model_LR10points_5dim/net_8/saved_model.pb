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
3feed_forward_sub_net_8/batch_normalization_48/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*D
shared_name53feed_forward_sub_net_8/batch_normalization_48/gamma
�
Gfeed_forward_sub_net_8/batch_normalization_48/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_8/batch_normalization_48/gamma*
_output_shapes
:
*
dtype0
�
2feed_forward_sub_net_8/batch_normalization_48/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*C
shared_name42feed_forward_sub_net_8/batch_normalization_48/beta
�
Ffeed_forward_sub_net_8/batch_normalization_48/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_8/batch_normalization_48/beta*
_output_shapes
:
*
dtype0
�
3feed_forward_sub_net_8/batch_normalization_49/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_8/batch_normalization_49/gamma
�
Gfeed_forward_sub_net_8/batch_normalization_49/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_8/batch_normalization_49/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_8/batch_normalization_49/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_8/batch_normalization_49/beta
�
Ffeed_forward_sub_net_8/batch_normalization_49/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_8/batch_normalization_49/beta*
_output_shapes
:*
dtype0
�
3feed_forward_sub_net_8/batch_normalization_50/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_8/batch_normalization_50/gamma
�
Gfeed_forward_sub_net_8/batch_normalization_50/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_8/batch_normalization_50/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_8/batch_normalization_50/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_8/batch_normalization_50/beta
�
Ffeed_forward_sub_net_8/batch_normalization_50/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_8/batch_normalization_50/beta*
_output_shapes
:*
dtype0
�
3feed_forward_sub_net_8/batch_normalization_51/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_8/batch_normalization_51/gamma
�
Gfeed_forward_sub_net_8/batch_normalization_51/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_8/batch_normalization_51/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_8/batch_normalization_51/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_8/batch_normalization_51/beta
�
Ffeed_forward_sub_net_8/batch_normalization_51/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_8/batch_normalization_51/beta*
_output_shapes
:*
dtype0
�
3feed_forward_sub_net_8/batch_normalization_52/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_8/batch_normalization_52/gamma
�
Gfeed_forward_sub_net_8/batch_normalization_52/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_8/batch_normalization_52/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_8/batch_normalization_52/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_8/batch_normalization_52/beta
�
Ffeed_forward_sub_net_8/batch_normalization_52/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_8/batch_normalization_52/beta*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_8/batch_normalization_48/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*J
shared_name;9feed_forward_sub_net_8/batch_normalization_48/moving_mean
�
Mfeed_forward_sub_net_8/batch_normalization_48/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_8/batch_normalization_48/moving_mean*
_output_shapes
:
*
dtype0
�
=feed_forward_sub_net_8/batch_normalization_48/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*N
shared_name?=feed_forward_sub_net_8/batch_normalization_48/moving_variance
�
Qfeed_forward_sub_net_8/batch_normalization_48/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_8/batch_normalization_48/moving_variance*
_output_shapes
:
*
dtype0
�
9feed_forward_sub_net_8/batch_normalization_49/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_8/batch_normalization_49/moving_mean
�
Mfeed_forward_sub_net_8/batch_normalization_49/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_8/batch_normalization_49/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_8/batch_normalization_49/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_8/batch_normalization_49/moving_variance
�
Qfeed_forward_sub_net_8/batch_normalization_49/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_8/batch_normalization_49/moving_variance*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_8/batch_normalization_50/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_8/batch_normalization_50/moving_mean
�
Mfeed_forward_sub_net_8/batch_normalization_50/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_8/batch_normalization_50/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_8/batch_normalization_50/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_8/batch_normalization_50/moving_variance
�
Qfeed_forward_sub_net_8/batch_normalization_50/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_8/batch_normalization_50/moving_variance*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_8/batch_normalization_51/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_8/batch_normalization_51/moving_mean
�
Mfeed_forward_sub_net_8/batch_normalization_51/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_8/batch_normalization_51/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_8/batch_normalization_51/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_8/batch_normalization_51/moving_variance
�
Qfeed_forward_sub_net_8/batch_normalization_51/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_8/batch_normalization_51/moving_variance*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_8/batch_normalization_52/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_8/batch_normalization_52/moving_mean
�
Mfeed_forward_sub_net_8/batch_normalization_52/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_8/batch_normalization_52/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_8/batch_normalization_52/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_8/batch_normalization_52/moving_variance
�
Qfeed_forward_sub_net_8/batch_normalization_52/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_8/batch_normalization_52/moving_variance*
_output_shapes
:*
dtype0
�
&feed_forward_sub_net_8/dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&feed_forward_sub_net_8/dense_40/kernel
�
:feed_forward_sub_net_8/dense_40/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_8/dense_40/kernel*
_output_shapes

:
*
dtype0
�
&feed_forward_sub_net_8/dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_8/dense_41/kernel
�
:feed_forward_sub_net_8/dense_41/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_8/dense_41/kernel*
_output_shapes

:*
dtype0
�
&feed_forward_sub_net_8/dense_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_8/dense_42/kernel
�
:feed_forward_sub_net_8/dense_42/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_8/dense_42/kernel*
_output_shapes

:*
dtype0
�
&feed_forward_sub_net_8/dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_8/dense_43/kernel
�
:feed_forward_sub_net_8/dense_43/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_8/dense_43/kernel*
_output_shapes

:*
dtype0
�
&feed_forward_sub_net_8/dense_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&feed_forward_sub_net_8/dense_44/kernel
�
:feed_forward_sub_net_8/dense_44/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_8/dense_44/kernel*
_output_shapes

:
*
dtype0
�
$feed_forward_sub_net_8/dense_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$feed_forward_sub_net_8/dense_44/bias
�
8feed_forward_sub_net_8/dense_44/bias/Read/ReadVariableOpReadVariableOp$feed_forward_sub_net_8/dense_44/bias*
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
VARIABLE_VALUE3feed_forward_sub_net_8/batch_normalization_48/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_8/batch_normalization_48/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_8/batch_normalization_49/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_8/batch_normalization_49/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_8/batch_normalization_50/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_8/batch_normalization_50/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_8/batch_normalization_51/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_8/batch_normalization_51/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_8/batch_normalization_52/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_8/batch_normalization_52/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_8/batch_normalization_48/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_8/batch_normalization_48/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_8/batch_normalization_49/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_8/batch_normalization_49/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_8/batch_normalization_50/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_8/batch_normalization_50/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_8/batch_normalization_51/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_8/batch_normalization_51/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_8/batch_normalization_52/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_8/batch_normalization_52/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_8/dense_40/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_8/dense_41/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_8/dense_42/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_8/dense_43/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_8/dense_44/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$feed_forward_sub_net_8/dense_44/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_19feed_forward_sub_net_8/batch_normalization_48/moving_mean=feed_forward_sub_net_8/batch_normalization_48/moving_variance2feed_forward_sub_net_8/batch_normalization_48/beta3feed_forward_sub_net_8/batch_normalization_48/gamma&feed_forward_sub_net_8/dense_40/kernel9feed_forward_sub_net_8/batch_normalization_49/moving_mean=feed_forward_sub_net_8/batch_normalization_49/moving_variance2feed_forward_sub_net_8/batch_normalization_49/beta3feed_forward_sub_net_8/batch_normalization_49/gamma&feed_forward_sub_net_8/dense_41/kernel9feed_forward_sub_net_8/batch_normalization_50/moving_mean=feed_forward_sub_net_8/batch_normalization_50/moving_variance2feed_forward_sub_net_8/batch_normalization_50/beta3feed_forward_sub_net_8/batch_normalization_50/gamma&feed_forward_sub_net_8/dense_42/kernel9feed_forward_sub_net_8/batch_normalization_51/moving_mean=feed_forward_sub_net_8/batch_normalization_51/moving_variance2feed_forward_sub_net_8/batch_normalization_51/beta3feed_forward_sub_net_8/batch_normalization_51/gamma&feed_forward_sub_net_8/dense_43/kernel9feed_forward_sub_net_8/batch_normalization_52/moving_mean=feed_forward_sub_net_8/batch_normalization_52/moving_variance2feed_forward_sub_net_8/batch_normalization_52/beta3feed_forward_sub_net_8/batch_normalization_52/gamma&feed_forward_sub_net_8/dense_44/kernel$feed_forward_sub_net_8/dense_44/bias*&
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
$__inference_signature_wrapper_329327
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameGfeed_forward_sub_net_8/batch_normalization_48/gamma/Read/ReadVariableOpFfeed_forward_sub_net_8/batch_normalization_48/beta/Read/ReadVariableOpGfeed_forward_sub_net_8/batch_normalization_49/gamma/Read/ReadVariableOpFfeed_forward_sub_net_8/batch_normalization_49/beta/Read/ReadVariableOpGfeed_forward_sub_net_8/batch_normalization_50/gamma/Read/ReadVariableOpFfeed_forward_sub_net_8/batch_normalization_50/beta/Read/ReadVariableOpGfeed_forward_sub_net_8/batch_normalization_51/gamma/Read/ReadVariableOpFfeed_forward_sub_net_8/batch_normalization_51/beta/Read/ReadVariableOpGfeed_forward_sub_net_8/batch_normalization_52/gamma/Read/ReadVariableOpFfeed_forward_sub_net_8/batch_normalization_52/beta/Read/ReadVariableOpMfeed_forward_sub_net_8/batch_normalization_48/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_8/batch_normalization_48/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_8/batch_normalization_49/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_8/batch_normalization_49/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_8/batch_normalization_50/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_8/batch_normalization_50/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_8/batch_normalization_51/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_8/batch_normalization_51/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_8/batch_normalization_52/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_8/batch_normalization_52/moving_variance/Read/ReadVariableOp:feed_forward_sub_net_8/dense_40/kernel/Read/ReadVariableOp:feed_forward_sub_net_8/dense_41/kernel/Read/ReadVariableOp:feed_forward_sub_net_8/dense_42/kernel/Read/ReadVariableOp:feed_forward_sub_net_8/dense_43/kernel/Read/ReadVariableOp:feed_forward_sub_net_8/dense_44/kernel/Read/ReadVariableOp8feed_forward_sub_net_8/dense_44/bias/Read/ReadVariableOpConst*'
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
__inference__traced_save_330725
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename3feed_forward_sub_net_8/batch_normalization_48/gamma2feed_forward_sub_net_8/batch_normalization_48/beta3feed_forward_sub_net_8/batch_normalization_49/gamma2feed_forward_sub_net_8/batch_normalization_49/beta3feed_forward_sub_net_8/batch_normalization_50/gamma2feed_forward_sub_net_8/batch_normalization_50/beta3feed_forward_sub_net_8/batch_normalization_51/gamma2feed_forward_sub_net_8/batch_normalization_51/beta3feed_forward_sub_net_8/batch_normalization_52/gamma2feed_forward_sub_net_8/batch_normalization_52/beta9feed_forward_sub_net_8/batch_normalization_48/moving_mean=feed_forward_sub_net_8/batch_normalization_48/moving_variance9feed_forward_sub_net_8/batch_normalization_49/moving_mean=feed_forward_sub_net_8/batch_normalization_49/moving_variance9feed_forward_sub_net_8/batch_normalization_50/moving_mean=feed_forward_sub_net_8/batch_normalization_50/moving_variance9feed_forward_sub_net_8/batch_normalization_51/moving_mean=feed_forward_sub_net_8/batch_normalization_51/moving_variance9feed_forward_sub_net_8/batch_normalization_52/moving_mean=feed_forward_sub_net_8/batch_normalization_52/moving_variance&feed_forward_sub_net_8/dense_40/kernel&feed_forward_sub_net_8/dense_41/kernel&feed_forward_sub_net_8/dense_42/kernel&feed_forward_sub_net_8/dense_43/kernel&feed_forward_sub_net_8/dense_44/kernel$feed_forward_sub_net_8/dense_44/bias*&
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
"__inference__traced_restore_330813��
�+
�
R__inference_batch_normalization_49_layer_call_and_return_conditional_losses_328095

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
7__inference_batch_normalization_49_layer_call_fn_330290

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
R__inference_batch_normalization_49_layer_call_and_return_conditional_losses_3280332
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
D__inference_dense_42_layer_call_and_return_conditional_losses_328738

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
�

�
D__inference_dense_44_layer_call_and_return_conditional_losses_330615

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
D__inference_dense_41_layer_call_and_return_conditional_losses_328717

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
�+
�
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_328427

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
D__inference_dense_43_layer_call_and_return_conditional_losses_330598

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
7__inference_batch_normalization_48_layer_call_fn_330208

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
R__inference_batch_normalization_48_layer_call_and_return_conditional_losses_3278672
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
$__inference_signature_wrapper_329327
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
!__inference__wrapped_model_3278432
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
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_330487

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
��
�
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329433
xA
3batch_normalization_48_cast_readvariableop_resource:
C
5batch_normalization_48_cast_1_readvariableop_resource:
C
5batch_normalization_48_cast_2_readvariableop_resource:
C
5batch_normalization_48_cast_3_readvariableop_resource:
9
'dense_40_matmul_readvariableop_resource:
A
3batch_normalization_49_cast_readvariableop_resource:C
5batch_normalization_49_cast_1_readvariableop_resource:C
5batch_normalization_49_cast_2_readvariableop_resource:C
5batch_normalization_49_cast_3_readvariableop_resource:9
'dense_41_matmul_readvariableop_resource:A
3batch_normalization_50_cast_readvariableop_resource:C
5batch_normalization_50_cast_1_readvariableop_resource:C
5batch_normalization_50_cast_2_readvariableop_resource:C
5batch_normalization_50_cast_3_readvariableop_resource:9
'dense_42_matmul_readvariableop_resource:A
3batch_normalization_51_cast_readvariableop_resource:C
5batch_normalization_51_cast_1_readvariableop_resource:C
5batch_normalization_51_cast_2_readvariableop_resource:C
5batch_normalization_51_cast_3_readvariableop_resource:9
'dense_43_matmul_readvariableop_resource:A
3batch_normalization_52_cast_readvariableop_resource:C
5batch_normalization_52_cast_1_readvariableop_resource:C
5batch_normalization_52_cast_2_readvariableop_resource:C
5batch_normalization_52_cast_3_readvariableop_resource:9
'dense_44_matmul_readvariableop_resource:
6
(dense_44_biasadd_readvariableop_resource:

identity��*batch_normalization_48/Cast/ReadVariableOp�,batch_normalization_48/Cast_1/ReadVariableOp�,batch_normalization_48/Cast_2/ReadVariableOp�,batch_normalization_48/Cast_3/ReadVariableOp�*batch_normalization_49/Cast/ReadVariableOp�,batch_normalization_49/Cast_1/ReadVariableOp�,batch_normalization_49/Cast_2/ReadVariableOp�,batch_normalization_49/Cast_3/ReadVariableOp�*batch_normalization_50/Cast/ReadVariableOp�,batch_normalization_50/Cast_1/ReadVariableOp�,batch_normalization_50/Cast_2/ReadVariableOp�,batch_normalization_50/Cast_3/ReadVariableOp�*batch_normalization_51/Cast/ReadVariableOp�,batch_normalization_51/Cast_1/ReadVariableOp�,batch_normalization_51/Cast_2/ReadVariableOp�,batch_normalization_51/Cast_3/ReadVariableOp�*batch_normalization_52/Cast/ReadVariableOp�,batch_normalization_52/Cast_1/ReadVariableOp�,batch_normalization_52/Cast_2/ReadVariableOp�,batch_normalization_52/Cast_3/ReadVariableOp�dense_40/MatMul/ReadVariableOp�dense_41/MatMul/ReadVariableOp�dense_42/MatMul/ReadVariableOp�dense_43/MatMul/ReadVariableOp�dense_44/BiasAdd/ReadVariableOp�dense_44/MatMul/ReadVariableOp�
*batch_normalization_48/Cast/ReadVariableOpReadVariableOp3batch_normalization_48_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_48/Cast/ReadVariableOp�
,batch_normalization_48/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_48_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_48/Cast_1/ReadVariableOp�
,batch_normalization_48/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_48_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_48/Cast_2/ReadVariableOp�
,batch_normalization_48/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_48_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_48/Cast_3/ReadVariableOp�
&batch_normalization_48/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_48/batchnorm/add/y�
$batch_normalization_48/batchnorm/addAddV24batch_normalization_48/Cast_1/ReadVariableOp:value:0/batch_normalization_48/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_48/batchnorm/add�
&batch_normalization_48/batchnorm/RsqrtRsqrt(batch_normalization_48/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_48/batchnorm/Rsqrt�
$batch_normalization_48/batchnorm/mulMul*batch_normalization_48/batchnorm/Rsqrt:y:04batch_normalization_48/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_48/batchnorm/mul�
&batch_normalization_48/batchnorm/mul_1Mulx(batch_normalization_48/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_48/batchnorm/mul_1�
&batch_normalization_48/batchnorm/mul_2Mul2batch_normalization_48/Cast/ReadVariableOp:value:0(batch_normalization_48/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_48/batchnorm/mul_2�
$batch_normalization_48/batchnorm/subSub4batch_normalization_48/Cast_2/ReadVariableOp:value:0*batch_normalization_48/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_48/batchnorm/sub�
&batch_normalization_48/batchnorm/add_1AddV2*batch_normalization_48/batchnorm/mul_1:z:0(batch_normalization_48/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_48/batchnorm/add_1�
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_40/MatMul/ReadVariableOp�
dense_40/MatMulMatMul*batch_normalization_48/batchnorm/add_1:z:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_40/MatMul�
*batch_normalization_49/Cast/ReadVariableOpReadVariableOp3batch_normalization_49_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_49/Cast/ReadVariableOp�
,batch_normalization_49/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_49_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_49/Cast_1/ReadVariableOp�
,batch_normalization_49/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_49_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_49/Cast_2/ReadVariableOp�
,batch_normalization_49/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_49_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_49/Cast_3/ReadVariableOp�
&batch_normalization_49/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_49/batchnorm/add/y�
$batch_normalization_49/batchnorm/addAddV24batch_normalization_49/Cast_1/ReadVariableOp:value:0/batch_normalization_49/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/add�
&batch_normalization_49/batchnorm/RsqrtRsqrt(batch_normalization_49/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_49/batchnorm/Rsqrt�
$batch_normalization_49/batchnorm/mulMul*batch_normalization_49/batchnorm/Rsqrt:y:04batch_normalization_49/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/mul�
&batch_normalization_49/batchnorm/mul_1Muldense_40/MatMul:product:0(batch_normalization_49/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_49/batchnorm/mul_1�
&batch_normalization_49/batchnorm/mul_2Mul2batch_normalization_49/Cast/ReadVariableOp:value:0(batch_normalization_49/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_49/batchnorm/mul_2�
$batch_normalization_49/batchnorm/subSub4batch_normalization_49/Cast_2/ReadVariableOp:value:0*batch_normalization_49/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/sub�
&batch_normalization_49/batchnorm/add_1AddV2*batch_normalization_49/batchnorm/mul_1:z:0(batch_normalization_49/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_49/batchnorm/add_1r
ReluRelu*batch_normalization_49/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_41/MatMul/ReadVariableOp�
dense_41/MatMulMatMulRelu:activations:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_41/MatMul�
*batch_normalization_50/Cast/ReadVariableOpReadVariableOp3batch_normalization_50_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_50/Cast/ReadVariableOp�
,batch_normalization_50/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_50_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_50/Cast_1/ReadVariableOp�
,batch_normalization_50/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_50_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_50/Cast_2/ReadVariableOp�
,batch_normalization_50/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_50_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_50/Cast_3/ReadVariableOp�
&batch_normalization_50/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_50/batchnorm/add/y�
$batch_normalization_50/batchnorm/addAddV24batch_normalization_50/Cast_1/ReadVariableOp:value:0/batch_normalization_50/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/add�
&batch_normalization_50/batchnorm/RsqrtRsqrt(batch_normalization_50/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_50/batchnorm/Rsqrt�
$batch_normalization_50/batchnorm/mulMul*batch_normalization_50/batchnorm/Rsqrt:y:04batch_normalization_50/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/mul�
&batch_normalization_50/batchnorm/mul_1Muldense_41/MatMul:product:0(batch_normalization_50/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_50/batchnorm/mul_1�
&batch_normalization_50/batchnorm/mul_2Mul2batch_normalization_50/Cast/ReadVariableOp:value:0(batch_normalization_50/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_50/batchnorm/mul_2�
$batch_normalization_50/batchnorm/subSub4batch_normalization_50/Cast_2/ReadVariableOp:value:0*batch_normalization_50/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/sub�
&batch_normalization_50/batchnorm/add_1AddV2*batch_normalization_50/batchnorm/mul_1:z:0(batch_normalization_50/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_50/batchnorm/add_1v
Relu_1Relu*batch_normalization_50/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_42/MatMul/ReadVariableOp�
dense_42/MatMulMatMulRelu_1:activations:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_42/MatMul�
*batch_normalization_51/Cast/ReadVariableOpReadVariableOp3batch_normalization_51_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_51/Cast/ReadVariableOp�
,batch_normalization_51/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_51_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_51/Cast_1/ReadVariableOp�
,batch_normalization_51/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_51_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_51/Cast_2/ReadVariableOp�
,batch_normalization_51/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_51_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_51/Cast_3/ReadVariableOp�
&batch_normalization_51/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_51/batchnorm/add/y�
$batch_normalization_51/batchnorm/addAddV24batch_normalization_51/Cast_1/ReadVariableOp:value:0/batch_normalization_51/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/add�
&batch_normalization_51/batchnorm/RsqrtRsqrt(batch_normalization_51/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_51/batchnorm/Rsqrt�
$batch_normalization_51/batchnorm/mulMul*batch_normalization_51/batchnorm/Rsqrt:y:04batch_normalization_51/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/mul�
&batch_normalization_51/batchnorm/mul_1Muldense_42/MatMul:product:0(batch_normalization_51/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_51/batchnorm/mul_1�
&batch_normalization_51/batchnorm/mul_2Mul2batch_normalization_51/Cast/ReadVariableOp:value:0(batch_normalization_51/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_51/batchnorm/mul_2�
$batch_normalization_51/batchnorm/subSub4batch_normalization_51/Cast_2/ReadVariableOp:value:0*batch_normalization_51/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/sub�
&batch_normalization_51/batchnorm/add_1AddV2*batch_normalization_51/batchnorm/mul_1:z:0(batch_normalization_51/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_51/batchnorm/add_1v
Relu_2Relu*batch_normalization_51/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_43/MatMul/ReadVariableOp�
dense_43/MatMulMatMulRelu_2:activations:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_43/MatMul�
*batch_normalization_52/Cast/ReadVariableOpReadVariableOp3batch_normalization_52_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_52/Cast/ReadVariableOp�
,batch_normalization_52/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_52_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_52/Cast_1/ReadVariableOp�
,batch_normalization_52/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_52_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_52/Cast_2/ReadVariableOp�
,batch_normalization_52/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_52_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_52/Cast_3/ReadVariableOp�
&batch_normalization_52/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_52/batchnorm/add/y�
$batch_normalization_52/batchnorm/addAddV24batch_normalization_52/Cast_1/ReadVariableOp:value:0/batch_normalization_52/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_52/batchnorm/add�
&batch_normalization_52/batchnorm/RsqrtRsqrt(batch_normalization_52/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_52/batchnorm/Rsqrt�
$batch_normalization_52/batchnorm/mulMul*batch_normalization_52/batchnorm/Rsqrt:y:04batch_normalization_52/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_52/batchnorm/mul�
&batch_normalization_52/batchnorm/mul_1Muldense_43/MatMul:product:0(batch_normalization_52/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_52/batchnorm/mul_1�
&batch_normalization_52/batchnorm/mul_2Mul2batch_normalization_52/Cast/ReadVariableOp:value:0(batch_normalization_52/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_52/batchnorm/mul_2�
$batch_normalization_52/batchnorm/subSub4batch_normalization_52/Cast_2/ReadVariableOp:value:0*batch_normalization_52/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_52/batchnorm/sub�
&batch_normalization_52/batchnorm/add_1AddV2*batch_normalization_52/batchnorm/mul_1:z:0(batch_normalization_52/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_52/batchnorm/add_1v
Relu_3Relu*batch_normalization_52/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_44/MatMul/ReadVariableOp�
dense_44/MatMulMatMulRelu_3:activations:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_44/MatMul�
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_44/BiasAdd/ReadVariableOp�
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_44/BiasAddt
IdentityIdentitydense_44/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�	
NoOpNoOp+^batch_normalization_48/Cast/ReadVariableOp-^batch_normalization_48/Cast_1/ReadVariableOp-^batch_normalization_48/Cast_2/ReadVariableOp-^batch_normalization_48/Cast_3/ReadVariableOp+^batch_normalization_49/Cast/ReadVariableOp-^batch_normalization_49/Cast_1/ReadVariableOp-^batch_normalization_49/Cast_2/ReadVariableOp-^batch_normalization_49/Cast_3/ReadVariableOp+^batch_normalization_50/Cast/ReadVariableOp-^batch_normalization_50/Cast_1/ReadVariableOp-^batch_normalization_50/Cast_2/ReadVariableOp-^batch_normalization_50/Cast_3/ReadVariableOp+^batch_normalization_51/Cast/ReadVariableOp-^batch_normalization_51/Cast_1/ReadVariableOp-^batch_normalization_51/Cast_2/ReadVariableOp-^batch_normalization_51/Cast_3/ReadVariableOp+^batch_normalization_52/Cast/ReadVariableOp-^batch_normalization_52/Cast_1/ReadVariableOp-^batch_normalization_52/Cast_2/ReadVariableOp-^batch_normalization_52/Cast_3/ReadVariableOp^dense_40/MatMul/ReadVariableOp^dense_41/MatMul/ReadVariableOp^dense_42/MatMul/ReadVariableOp^dense_43/MatMul/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_48/Cast/ReadVariableOp*batch_normalization_48/Cast/ReadVariableOp2\
,batch_normalization_48/Cast_1/ReadVariableOp,batch_normalization_48/Cast_1/ReadVariableOp2\
,batch_normalization_48/Cast_2/ReadVariableOp,batch_normalization_48/Cast_2/ReadVariableOp2\
,batch_normalization_48/Cast_3/ReadVariableOp,batch_normalization_48/Cast_3/ReadVariableOp2X
*batch_normalization_49/Cast/ReadVariableOp*batch_normalization_49/Cast/ReadVariableOp2\
,batch_normalization_49/Cast_1/ReadVariableOp,batch_normalization_49/Cast_1/ReadVariableOp2\
,batch_normalization_49/Cast_2/ReadVariableOp,batch_normalization_49/Cast_2/ReadVariableOp2\
,batch_normalization_49/Cast_3/ReadVariableOp,batch_normalization_49/Cast_3/ReadVariableOp2X
*batch_normalization_50/Cast/ReadVariableOp*batch_normalization_50/Cast/ReadVariableOp2\
,batch_normalization_50/Cast_1/ReadVariableOp,batch_normalization_50/Cast_1/ReadVariableOp2\
,batch_normalization_50/Cast_2/ReadVariableOp,batch_normalization_50/Cast_2/ReadVariableOp2\
,batch_normalization_50/Cast_3/ReadVariableOp,batch_normalization_50/Cast_3/ReadVariableOp2X
*batch_normalization_51/Cast/ReadVariableOp*batch_normalization_51/Cast/ReadVariableOp2\
,batch_normalization_51/Cast_1/ReadVariableOp,batch_normalization_51/Cast_1/ReadVariableOp2\
,batch_normalization_51/Cast_2/ReadVariableOp,batch_normalization_51/Cast_2/ReadVariableOp2\
,batch_normalization_51/Cast_3/ReadVariableOp,batch_normalization_51/Cast_3/ReadVariableOp2X
*batch_normalization_52/Cast/ReadVariableOp*batch_normalization_52/Cast/ReadVariableOp2\
,batch_normalization_52/Cast_1/ReadVariableOp,batch_normalization_52/Cast_1/ReadVariableOp2\
,batch_normalization_52/Cast_2/ReadVariableOp,batch_normalization_52/Cast_2/ReadVariableOp2\
,batch_normalization_52/Cast_3/ReadVariableOp,batch_normalization_52/Cast_3/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������


_user_specified_namex
�D
�
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_328790
x+
batch_normalization_48_328680:
+
batch_normalization_48_328682:
+
batch_normalization_48_328684:
+
batch_normalization_48_328686:
!
dense_40_328697:
+
batch_normalization_49_328700:+
batch_normalization_49_328702:+
batch_normalization_49_328704:+
batch_normalization_49_328706:!
dense_41_328718:+
batch_normalization_50_328721:+
batch_normalization_50_328723:+
batch_normalization_50_328725:+
batch_normalization_50_328727:!
dense_42_328739:+
batch_normalization_51_328742:+
batch_normalization_51_328744:+
batch_normalization_51_328746:+
batch_normalization_51_328748:!
dense_43_328760:+
batch_normalization_52_328763:+
batch_normalization_52_328765:+
batch_normalization_52_328767:+
batch_normalization_52_328769:!
dense_44_328784:

dense_44_328786:

identity��.batch_normalization_48/StatefulPartitionedCall�.batch_normalization_49/StatefulPartitionedCall�.batch_normalization_50/StatefulPartitionedCall�.batch_normalization_51/StatefulPartitionedCall�.batch_normalization_52/StatefulPartitionedCall� dense_40/StatefulPartitionedCall� dense_41/StatefulPartitionedCall� dense_42/StatefulPartitionedCall� dense_43/StatefulPartitionedCall� dense_44/StatefulPartitionedCall�
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_48_328680batch_normalization_48_328682batch_normalization_48_328684batch_normalization_48_328686*
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
R__inference_batch_normalization_48_layer_call_and_return_conditional_losses_32786720
.batch_normalization_48/StatefulPartitionedCall�
 dense_40/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_48/StatefulPartitionedCall:output:0dense_40_328697*
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
D__inference_dense_40_layer_call_and_return_conditional_losses_3286962"
 dense_40/StatefulPartitionedCall�
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0batch_normalization_49_328700batch_normalization_49_328702batch_normalization_49_328704batch_normalization_49_328706*
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
R__inference_batch_normalization_49_layer_call_and_return_conditional_losses_32803320
.batch_normalization_49/StatefulPartitionedCall
ReluRelu7batch_normalization_49/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu�
 dense_41/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_41_328718*
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
D__inference_dense_41_layer_call_and_return_conditional_losses_3287172"
 dense_41/StatefulPartitionedCall�
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0batch_normalization_50_328721batch_normalization_50_328723batch_normalization_50_328725batch_normalization_50_328727*
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
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_32819920
.batch_normalization_50/StatefulPartitionedCall�
Relu_1Relu7batch_normalization_50/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_1�
 dense_42/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_42_328739*
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
D__inference_dense_42_layer_call_and_return_conditional_losses_3287382"
 dense_42/StatefulPartitionedCall�
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0batch_normalization_51_328742batch_normalization_51_328744batch_normalization_51_328746batch_normalization_51_328748*
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
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_32836520
.batch_normalization_51/StatefulPartitionedCall�
Relu_2Relu7batch_normalization_51/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_2�
 dense_43/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_43_328760*
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
D__inference_dense_43_layer_call_and_return_conditional_losses_3287592"
 dense_43/StatefulPartitionedCall�
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:0batch_normalization_52_328763batch_normalization_52_328765batch_normalization_52_328767batch_normalization_52_328769*
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
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_32853120
.batch_normalization_52/StatefulPartitionedCall�
Relu_3Relu7batch_normalization_52/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_3�
 dense_44/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_44_328784dense_44_328786*
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
D__inference_dense_44_layer_call_and_return_conditional_losses_3287832"
 dense_44/StatefulPartitionedCall�
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp/^batch_normalization_48/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_52/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall:J F
'
_output_shapes
:���������


_user_specified_namex
�D
�
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329016
x+
batch_normalization_48_328949:
+
batch_normalization_48_328951:
+
batch_normalization_48_328953:
+
batch_normalization_48_328955:
!
dense_40_328958:
+
batch_normalization_49_328961:+
batch_normalization_49_328963:+
batch_normalization_49_328965:+
batch_normalization_49_328967:!
dense_41_328971:+
batch_normalization_50_328974:+
batch_normalization_50_328976:+
batch_normalization_50_328978:+
batch_normalization_50_328980:!
dense_42_328984:+
batch_normalization_51_328987:+
batch_normalization_51_328989:+
batch_normalization_51_328991:+
batch_normalization_51_328993:!
dense_43_328997:+
batch_normalization_52_329000:+
batch_normalization_52_329002:+
batch_normalization_52_329004:+
batch_normalization_52_329006:!
dense_44_329010:

dense_44_329012:

identity��.batch_normalization_48/StatefulPartitionedCall�.batch_normalization_49/StatefulPartitionedCall�.batch_normalization_50/StatefulPartitionedCall�.batch_normalization_51/StatefulPartitionedCall�.batch_normalization_52/StatefulPartitionedCall� dense_40/StatefulPartitionedCall� dense_41/StatefulPartitionedCall� dense_42/StatefulPartitionedCall� dense_43/StatefulPartitionedCall� dense_44/StatefulPartitionedCall�
.batch_normalization_48/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_48_328949batch_normalization_48_328951batch_normalization_48_328953batch_normalization_48_328955*
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
R__inference_batch_normalization_48_layer_call_and_return_conditional_losses_32792920
.batch_normalization_48/StatefulPartitionedCall�
 dense_40/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_48/StatefulPartitionedCall:output:0dense_40_328958*
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
D__inference_dense_40_layer_call_and_return_conditional_losses_3286962"
 dense_40/StatefulPartitionedCall�
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0batch_normalization_49_328961batch_normalization_49_328963batch_normalization_49_328965batch_normalization_49_328967*
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
R__inference_batch_normalization_49_layer_call_and_return_conditional_losses_32809520
.batch_normalization_49/StatefulPartitionedCall
ReluRelu7batch_normalization_49/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu�
 dense_41/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_41_328971*
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
D__inference_dense_41_layer_call_and_return_conditional_losses_3287172"
 dense_41/StatefulPartitionedCall�
.batch_normalization_50/StatefulPartitionedCallStatefulPartitionedCall)dense_41/StatefulPartitionedCall:output:0batch_normalization_50_328974batch_normalization_50_328976batch_normalization_50_328978batch_normalization_50_328980*
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
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_32826120
.batch_normalization_50/StatefulPartitionedCall�
Relu_1Relu7batch_normalization_50/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_1�
 dense_42/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_42_328984*
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
D__inference_dense_42_layer_call_and_return_conditional_losses_3287382"
 dense_42/StatefulPartitionedCall�
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0batch_normalization_51_328987batch_normalization_51_328989batch_normalization_51_328991batch_normalization_51_328993*
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
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_32842720
.batch_normalization_51/StatefulPartitionedCall�
Relu_2Relu7batch_normalization_51/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_2�
 dense_43/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_43_328997*
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
D__inference_dense_43_layer_call_and_return_conditional_losses_3287592"
 dense_43/StatefulPartitionedCall�
.batch_normalization_52/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:0batch_normalization_52_329000batch_normalization_52_329002batch_normalization_52_329004batch_normalization_52_329006*
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
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_32859320
.batch_normalization_52/StatefulPartitionedCall�
Relu_3Relu7batch_normalization_52/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_3�
 dense_44/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_44_329010dense_44_329012*
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
D__inference_dense_44_layer_call_and_return_conditional_losses_3287832"
 dense_44/StatefulPartitionedCall�
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp/^batch_normalization_48/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_50/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_52/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_48/StatefulPartitionedCall.batch_normalization_48/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_50/StatefulPartitionedCall.batch_normalization_50/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_52/StatefulPartitionedCall.batch_normalization_52/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall:J F
'
_output_shapes
:���������


_user_specified_namex
�
�
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_328199

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
7__inference_feed_forward_sub_net_8_layer_call_fn_329968
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
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_3287902
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
D__inference_dense_43_layer_call_and_return_conditional_losses_328759

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
�
�
7__inference_feed_forward_sub_net_8_layer_call_fn_330025
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
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_3287902
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
�+
�
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_328261

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
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_328531

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
�

�
D__inference_dense_44_layer_call_and_return_conditional_losses_328783

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
��
�
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329619
xL
>batch_normalization_48_assignmovingavg_readvariableop_resource:
N
@batch_normalization_48_assignmovingavg_1_readvariableop_resource:
A
3batch_normalization_48_cast_readvariableop_resource:
C
5batch_normalization_48_cast_1_readvariableop_resource:
9
'dense_40_matmul_readvariableop_resource:
L
>batch_normalization_49_assignmovingavg_readvariableop_resource:N
@batch_normalization_49_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_49_cast_readvariableop_resource:C
5batch_normalization_49_cast_1_readvariableop_resource:9
'dense_41_matmul_readvariableop_resource:L
>batch_normalization_50_assignmovingavg_readvariableop_resource:N
@batch_normalization_50_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_50_cast_readvariableop_resource:C
5batch_normalization_50_cast_1_readvariableop_resource:9
'dense_42_matmul_readvariableop_resource:L
>batch_normalization_51_assignmovingavg_readvariableop_resource:N
@batch_normalization_51_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_51_cast_readvariableop_resource:C
5batch_normalization_51_cast_1_readvariableop_resource:9
'dense_43_matmul_readvariableop_resource:L
>batch_normalization_52_assignmovingavg_readvariableop_resource:N
@batch_normalization_52_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_52_cast_readvariableop_resource:C
5batch_normalization_52_cast_1_readvariableop_resource:9
'dense_44_matmul_readvariableop_resource:
6
(dense_44_biasadd_readvariableop_resource:

identity��&batch_normalization_48/AssignMovingAvg�5batch_normalization_48/AssignMovingAvg/ReadVariableOp�(batch_normalization_48/AssignMovingAvg_1�7batch_normalization_48/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_48/Cast/ReadVariableOp�,batch_normalization_48/Cast_1/ReadVariableOp�&batch_normalization_49/AssignMovingAvg�5batch_normalization_49/AssignMovingAvg/ReadVariableOp�(batch_normalization_49/AssignMovingAvg_1�7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_49/Cast/ReadVariableOp�,batch_normalization_49/Cast_1/ReadVariableOp�&batch_normalization_50/AssignMovingAvg�5batch_normalization_50/AssignMovingAvg/ReadVariableOp�(batch_normalization_50/AssignMovingAvg_1�7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_50/Cast/ReadVariableOp�,batch_normalization_50/Cast_1/ReadVariableOp�&batch_normalization_51/AssignMovingAvg�5batch_normalization_51/AssignMovingAvg/ReadVariableOp�(batch_normalization_51/AssignMovingAvg_1�7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_51/Cast/ReadVariableOp�,batch_normalization_51/Cast_1/ReadVariableOp�&batch_normalization_52/AssignMovingAvg�5batch_normalization_52/AssignMovingAvg/ReadVariableOp�(batch_normalization_52/AssignMovingAvg_1�7batch_normalization_52/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_52/Cast/ReadVariableOp�,batch_normalization_52/Cast_1/ReadVariableOp�dense_40/MatMul/ReadVariableOp�dense_41/MatMul/ReadVariableOp�dense_42/MatMul/ReadVariableOp�dense_43/MatMul/ReadVariableOp�dense_44/BiasAdd/ReadVariableOp�dense_44/MatMul/ReadVariableOp�
5batch_normalization_48/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_48/moments/mean/reduction_indices�
#batch_normalization_48/moments/meanMeanx>batch_normalization_48/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_48/moments/mean�
+batch_normalization_48/moments/StopGradientStopGradient,batch_normalization_48/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_48/moments/StopGradient�
0batch_normalization_48/moments/SquaredDifferenceSquaredDifferencex4batch_normalization_48/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������
22
0batch_normalization_48/moments/SquaredDifference�
9batch_normalization_48/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_48/moments/variance/reduction_indices�
'batch_normalization_48/moments/varianceMean4batch_normalization_48/moments/SquaredDifference:z:0Bbatch_normalization_48/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_48/moments/variance�
&batch_normalization_48/moments/SqueezeSqueeze,batch_normalization_48/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_48/moments/Squeeze�
(batch_normalization_48/moments/Squeeze_1Squeeze0batch_normalization_48/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_48/moments/Squeeze_1�
,batch_normalization_48/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_48/AssignMovingAvg/decay�
+batch_normalization_48/AssignMovingAvg/CastCast5batch_normalization_48/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_48/AssignMovingAvg/Cast�
5batch_normalization_48/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_48_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_48/AssignMovingAvg/ReadVariableOp�
*batch_normalization_48/AssignMovingAvg/subSub=batch_normalization_48/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_48/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_48/AssignMovingAvg/sub�
*batch_normalization_48/AssignMovingAvg/mulMul.batch_normalization_48/AssignMovingAvg/sub:z:0/batch_normalization_48/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_48/AssignMovingAvg/mul�
&batch_normalization_48/AssignMovingAvgAssignSubVariableOp>batch_normalization_48_assignmovingavg_readvariableop_resource.batch_normalization_48/AssignMovingAvg/mul:z:06^batch_normalization_48/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_48/AssignMovingAvg�
.batch_normalization_48/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_48/AssignMovingAvg_1/decay�
-batch_normalization_48/AssignMovingAvg_1/CastCast7batch_normalization_48/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_48/AssignMovingAvg_1/Cast�
7batch_normalization_48/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_48_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_48/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_48/AssignMovingAvg_1/subSub?batch_normalization_48/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_48/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_48/AssignMovingAvg_1/sub�
,batch_normalization_48/AssignMovingAvg_1/mulMul0batch_normalization_48/AssignMovingAvg_1/sub:z:01batch_normalization_48/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_48/AssignMovingAvg_1/mul�
(batch_normalization_48/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_48_assignmovingavg_1_readvariableop_resource0batch_normalization_48/AssignMovingAvg_1/mul:z:08^batch_normalization_48/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_48/AssignMovingAvg_1�
*batch_normalization_48/Cast/ReadVariableOpReadVariableOp3batch_normalization_48_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_48/Cast/ReadVariableOp�
,batch_normalization_48/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_48_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_48/Cast_1/ReadVariableOp�
&batch_normalization_48/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_48/batchnorm/add/y�
$batch_normalization_48/batchnorm/addAddV21batch_normalization_48/moments/Squeeze_1:output:0/batch_normalization_48/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_48/batchnorm/add�
&batch_normalization_48/batchnorm/RsqrtRsqrt(batch_normalization_48/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_48/batchnorm/Rsqrt�
$batch_normalization_48/batchnorm/mulMul*batch_normalization_48/batchnorm/Rsqrt:y:04batch_normalization_48/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_48/batchnorm/mul�
&batch_normalization_48/batchnorm/mul_1Mulx(batch_normalization_48/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_48/batchnorm/mul_1�
&batch_normalization_48/batchnorm/mul_2Mul/batch_normalization_48/moments/Squeeze:output:0(batch_normalization_48/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_48/batchnorm/mul_2�
$batch_normalization_48/batchnorm/subSub2batch_normalization_48/Cast/ReadVariableOp:value:0*batch_normalization_48/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_48/batchnorm/sub�
&batch_normalization_48/batchnorm/add_1AddV2*batch_normalization_48/batchnorm/mul_1:z:0(batch_normalization_48/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_48/batchnorm/add_1�
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_40/MatMul/ReadVariableOp�
dense_40/MatMulMatMul*batch_normalization_48/batchnorm/add_1:z:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_40/MatMul�
5batch_normalization_49/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_49/moments/mean/reduction_indices�
#batch_normalization_49/moments/meanMeandense_40/MatMul:product:0>batch_normalization_49/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_49/moments/mean�
+batch_normalization_49/moments/StopGradientStopGradient,batch_normalization_49/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_49/moments/StopGradient�
0batch_normalization_49/moments/SquaredDifferenceSquaredDifferencedense_40/MatMul:product:04batch_normalization_49/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_49/moments/SquaredDifference�
9batch_normalization_49/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_49/moments/variance/reduction_indices�
'batch_normalization_49/moments/varianceMean4batch_normalization_49/moments/SquaredDifference:z:0Bbatch_normalization_49/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_49/moments/variance�
&batch_normalization_49/moments/SqueezeSqueeze,batch_normalization_49/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_49/moments/Squeeze�
(batch_normalization_49/moments/Squeeze_1Squeeze0batch_normalization_49/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_49/moments/Squeeze_1�
,batch_normalization_49/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_49/AssignMovingAvg/decay�
+batch_normalization_49/AssignMovingAvg/CastCast5batch_normalization_49/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_49/AssignMovingAvg/Cast�
5batch_normalization_49/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_49_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_49/AssignMovingAvg/ReadVariableOp�
*batch_normalization_49/AssignMovingAvg/subSub=batch_normalization_49/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_49/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_49/AssignMovingAvg/sub�
*batch_normalization_49/AssignMovingAvg/mulMul.batch_normalization_49/AssignMovingAvg/sub:z:0/batch_normalization_49/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_49/AssignMovingAvg/mul�
&batch_normalization_49/AssignMovingAvgAssignSubVariableOp>batch_normalization_49_assignmovingavg_readvariableop_resource.batch_normalization_49/AssignMovingAvg/mul:z:06^batch_normalization_49/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_49/AssignMovingAvg�
.batch_normalization_49/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_49/AssignMovingAvg_1/decay�
-batch_normalization_49/AssignMovingAvg_1/CastCast7batch_normalization_49/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_49/AssignMovingAvg_1/Cast�
7batch_normalization_49/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_49_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_49/AssignMovingAvg_1/subSub?batch_normalization_49/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_49/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_49/AssignMovingAvg_1/sub�
,batch_normalization_49/AssignMovingAvg_1/mulMul0batch_normalization_49/AssignMovingAvg_1/sub:z:01batch_normalization_49/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_49/AssignMovingAvg_1/mul�
(batch_normalization_49/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_49_assignmovingavg_1_readvariableop_resource0batch_normalization_49/AssignMovingAvg_1/mul:z:08^batch_normalization_49/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_49/AssignMovingAvg_1�
*batch_normalization_49/Cast/ReadVariableOpReadVariableOp3batch_normalization_49_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_49/Cast/ReadVariableOp�
,batch_normalization_49/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_49_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_49/Cast_1/ReadVariableOp�
&batch_normalization_49/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_49/batchnorm/add/y�
$batch_normalization_49/batchnorm/addAddV21batch_normalization_49/moments/Squeeze_1:output:0/batch_normalization_49/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/add�
&batch_normalization_49/batchnorm/RsqrtRsqrt(batch_normalization_49/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_49/batchnorm/Rsqrt�
$batch_normalization_49/batchnorm/mulMul*batch_normalization_49/batchnorm/Rsqrt:y:04batch_normalization_49/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/mul�
&batch_normalization_49/batchnorm/mul_1Muldense_40/MatMul:product:0(batch_normalization_49/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_49/batchnorm/mul_1�
&batch_normalization_49/batchnorm/mul_2Mul/batch_normalization_49/moments/Squeeze:output:0(batch_normalization_49/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_49/batchnorm/mul_2�
$batch_normalization_49/batchnorm/subSub2batch_normalization_49/Cast/ReadVariableOp:value:0*batch_normalization_49/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/sub�
&batch_normalization_49/batchnorm/add_1AddV2*batch_normalization_49/batchnorm/mul_1:z:0(batch_normalization_49/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_49/batchnorm/add_1r
ReluRelu*batch_normalization_49/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_41/MatMul/ReadVariableOp�
dense_41/MatMulMatMulRelu:activations:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_41/MatMul�
5batch_normalization_50/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_50/moments/mean/reduction_indices�
#batch_normalization_50/moments/meanMeandense_41/MatMul:product:0>batch_normalization_50/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_50/moments/mean�
+batch_normalization_50/moments/StopGradientStopGradient,batch_normalization_50/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_50/moments/StopGradient�
0batch_normalization_50/moments/SquaredDifferenceSquaredDifferencedense_41/MatMul:product:04batch_normalization_50/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_50/moments/SquaredDifference�
9batch_normalization_50/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_50/moments/variance/reduction_indices�
'batch_normalization_50/moments/varianceMean4batch_normalization_50/moments/SquaredDifference:z:0Bbatch_normalization_50/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_50/moments/variance�
&batch_normalization_50/moments/SqueezeSqueeze,batch_normalization_50/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_50/moments/Squeeze�
(batch_normalization_50/moments/Squeeze_1Squeeze0batch_normalization_50/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_50/moments/Squeeze_1�
,batch_normalization_50/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_50/AssignMovingAvg/decay�
+batch_normalization_50/AssignMovingAvg/CastCast5batch_normalization_50/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_50/AssignMovingAvg/Cast�
5batch_normalization_50/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_50_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_50/AssignMovingAvg/ReadVariableOp�
*batch_normalization_50/AssignMovingAvg/subSub=batch_normalization_50/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_50/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_50/AssignMovingAvg/sub�
*batch_normalization_50/AssignMovingAvg/mulMul.batch_normalization_50/AssignMovingAvg/sub:z:0/batch_normalization_50/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_50/AssignMovingAvg/mul�
&batch_normalization_50/AssignMovingAvgAssignSubVariableOp>batch_normalization_50_assignmovingavg_readvariableop_resource.batch_normalization_50/AssignMovingAvg/mul:z:06^batch_normalization_50/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_50/AssignMovingAvg�
.batch_normalization_50/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_50/AssignMovingAvg_1/decay�
-batch_normalization_50/AssignMovingAvg_1/CastCast7batch_normalization_50/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_50/AssignMovingAvg_1/Cast�
7batch_normalization_50/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_50_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_50/AssignMovingAvg_1/subSub?batch_normalization_50/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_50/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_50/AssignMovingAvg_1/sub�
,batch_normalization_50/AssignMovingAvg_1/mulMul0batch_normalization_50/AssignMovingAvg_1/sub:z:01batch_normalization_50/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_50/AssignMovingAvg_1/mul�
(batch_normalization_50/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_50_assignmovingavg_1_readvariableop_resource0batch_normalization_50/AssignMovingAvg_1/mul:z:08^batch_normalization_50/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_50/AssignMovingAvg_1�
*batch_normalization_50/Cast/ReadVariableOpReadVariableOp3batch_normalization_50_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_50/Cast/ReadVariableOp�
,batch_normalization_50/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_50_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_50/Cast_1/ReadVariableOp�
&batch_normalization_50/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_50/batchnorm/add/y�
$batch_normalization_50/batchnorm/addAddV21batch_normalization_50/moments/Squeeze_1:output:0/batch_normalization_50/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/add�
&batch_normalization_50/batchnorm/RsqrtRsqrt(batch_normalization_50/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_50/batchnorm/Rsqrt�
$batch_normalization_50/batchnorm/mulMul*batch_normalization_50/batchnorm/Rsqrt:y:04batch_normalization_50/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/mul�
&batch_normalization_50/batchnorm/mul_1Muldense_41/MatMul:product:0(batch_normalization_50/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_50/batchnorm/mul_1�
&batch_normalization_50/batchnorm/mul_2Mul/batch_normalization_50/moments/Squeeze:output:0(batch_normalization_50/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_50/batchnorm/mul_2�
$batch_normalization_50/batchnorm/subSub2batch_normalization_50/Cast/ReadVariableOp:value:0*batch_normalization_50/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/sub�
&batch_normalization_50/batchnorm/add_1AddV2*batch_normalization_50/batchnorm/mul_1:z:0(batch_normalization_50/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_50/batchnorm/add_1v
Relu_1Relu*batch_normalization_50/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_42/MatMul/ReadVariableOp�
dense_42/MatMulMatMulRelu_1:activations:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_42/MatMul�
5batch_normalization_51/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_51/moments/mean/reduction_indices�
#batch_normalization_51/moments/meanMeandense_42/MatMul:product:0>batch_normalization_51/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_51/moments/mean�
+batch_normalization_51/moments/StopGradientStopGradient,batch_normalization_51/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_51/moments/StopGradient�
0batch_normalization_51/moments/SquaredDifferenceSquaredDifferencedense_42/MatMul:product:04batch_normalization_51/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_51/moments/SquaredDifference�
9batch_normalization_51/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_51/moments/variance/reduction_indices�
'batch_normalization_51/moments/varianceMean4batch_normalization_51/moments/SquaredDifference:z:0Bbatch_normalization_51/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_51/moments/variance�
&batch_normalization_51/moments/SqueezeSqueeze,batch_normalization_51/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_51/moments/Squeeze�
(batch_normalization_51/moments/Squeeze_1Squeeze0batch_normalization_51/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_51/moments/Squeeze_1�
,batch_normalization_51/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_51/AssignMovingAvg/decay�
+batch_normalization_51/AssignMovingAvg/CastCast5batch_normalization_51/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_51/AssignMovingAvg/Cast�
5batch_normalization_51/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_51_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_51/AssignMovingAvg/ReadVariableOp�
*batch_normalization_51/AssignMovingAvg/subSub=batch_normalization_51/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_51/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_51/AssignMovingAvg/sub�
*batch_normalization_51/AssignMovingAvg/mulMul.batch_normalization_51/AssignMovingAvg/sub:z:0/batch_normalization_51/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_51/AssignMovingAvg/mul�
&batch_normalization_51/AssignMovingAvgAssignSubVariableOp>batch_normalization_51_assignmovingavg_readvariableop_resource.batch_normalization_51/AssignMovingAvg/mul:z:06^batch_normalization_51/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_51/AssignMovingAvg�
.batch_normalization_51/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_51/AssignMovingAvg_1/decay�
-batch_normalization_51/AssignMovingAvg_1/CastCast7batch_normalization_51/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_51/AssignMovingAvg_1/Cast�
7batch_normalization_51/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_51_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_51/AssignMovingAvg_1/subSub?batch_normalization_51/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_51/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_51/AssignMovingAvg_1/sub�
,batch_normalization_51/AssignMovingAvg_1/mulMul0batch_normalization_51/AssignMovingAvg_1/sub:z:01batch_normalization_51/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_51/AssignMovingAvg_1/mul�
(batch_normalization_51/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_51_assignmovingavg_1_readvariableop_resource0batch_normalization_51/AssignMovingAvg_1/mul:z:08^batch_normalization_51/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_51/AssignMovingAvg_1�
*batch_normalization_51/Cast/ReadVariableOpReadVariableOp3batch_normalization_51_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_51/Cast/ReadVariableOp�
,batch_normalization_51/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_51_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_51/Cast_1/ReadVariableOp�
&batch_normalization_51/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_51/batchnorm/add/y�
$batch_normalization_51/batchnorm/addAddV21batch_normalization_51/moments/Squeeze_1:output:0/batch_normalization_51/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/add�
&batch_normalization_51/batchnorm/RsqrtRsqrt(batch_normalization_51/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_51/batchnorm/Rsqrt�
$batch_normalization_51/batchnorm/mulMul*batch_normalization_51/batchnorm/Rsqrt:y:04batch_normalization_51/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/mul�
&batch_normalization_51/batchnorm/mul_1Muldense_42/MatMul:product:0(batch_normalization_51/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_51/batchnorm/mul_1�
&batch_normalization_51/batchnorm/mul_2Mul/batch_normalization_51/moments/Squeeze:output:0(batch_normalization_51/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_51/batchnorm/mul_2�
$batch_normalization_51/batchnorm/subSub2batch_normalization_51/Cast/ReadVariableOp:value:0*batch_normalization_51/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/sub�
&batch_normalization_51/batchnorm/add_1AddV2*batch_normalization_51/batchnorm/mul_1:z:0(batch_normalization_51/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_51/batchnorm/add_1v
Relu_2Relu*batch_normalization_51/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_43/MatMul/ReadVariableOp�
dense_43/MatMulMatMulRelu_2:activations:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_43/MatMul�
5batch_normalization_52/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_52/moments/mean/reduction_indices�
#batch_normalization_52/moments/meanMeandense_43/MatMul:product:0>batch_normalization_52/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_52/moments/mean�
+batch_normalization_52/moments/StopGradientStopGradient,batch_normalization_52/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_52/moments/StopGradient�
0batch_normalization_52/moments/SquaredDifferenceSquaredDifferencedense_43/MatMul:product:04batch_normalization_52/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_52/moments/SquaredDifference�
9batch_normalization_52/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_52/moments/variance/reduction_indices�
'batch_normalization_52/moments/varianceMean4batch_normalization_52/moments/SquaredDifference:z:0Bbatch_normalization_52/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_52/moments/variance�
&batch_normalization_52/moments/SqueezeSqueeze,batch_normalization_52/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_52/moments/Squeeze�
(batch_normalization_52/moments/Squeeze_1Squeeze0batch_normalization_52/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_52/moments/Squeeze_1�
,batch_normalization_52/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_52/AssignMovingAvg/decay�
+batch_normalization_52/AssignMovingAvg/CastCast5batch_normalization_52/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_52/AssignMovingAvg/Cast�
5batch_normalization_52/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_52_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_52/AssignMovingAvg/ReadVariableOp�
*batch_normalization_52/AssignMovingAvg/subSub=batch_normalization_52/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_52/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_52/AssignMovingAvg/sub�
*batch_normalization_52/AssignMovingAvg/mulMul.batch_normalization_52/AssignMovingAvg/sub:z:0/batch_normalization_52/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_52/AssignMovingAvg/mul�
&batch_normalization_52/AssignMovingAvgAssignSubVariableOp>batch_normalization_52_assignmovingavg_readvariableop_resource.batch_normalization_52/AssignMovingAvg/mul:z:06^batch_normalization_52/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_52/AssignMovingAvg�
.batch_normalization_52/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_52/AssignMovingAvg_1/decay�
-batch_normalization_52/AssignMovingAvg_1/CastCast7batch_normalization_52/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_52/AssignMovingAvg_1/Cast�
7batch_normalization_52/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_52_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_52/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_52/AssignMovingAvg_1/subSub?batch_normalization_52/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_52/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_52/AssignMovingAvg_1/sub�
,batch_normalization_52/AssignMovingAvg_1/mulMul0batch_normalization_52/AssignMovingAvg_1/sub:z:01batch_normalization_52/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_52/AssignMovingAvg_1/mul�
(batch_normalization_52/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_52_assignmovingavg_1_readvariableop_resource0batch_normalization_52/AssignMovingAvg_1/mul:z:08^batch_normalization_52/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_52/AssignMovingAvg_1�
*batch_normalization_52/Cast/ReadVariableOpReadVariableOp3batch_normalization_52_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_52/Cast/ReadVariableOp�
,batch_normalization_52/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_52_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_52/Cast_1/ReadVariableOp�
&batch_normalization_52/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_52/batchnorm/add/y�
$batch_normalization_52/batchnorm/addAddV21batch_normalization_52/moments/Squeeze_1:output:0/batch_normalization_52/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_52/batchnorm/add�
&batch_normalization_52/batchnorm/RsqrtRsqrt(batch_normalization_52/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_52/batchnorm/Rsqrt�
$batch_normalization_52/batchnorm/mulMul*batch_normalization_52/batchnorm/Rsqrt:y:04batch_normalization_52/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_52/batchnorm/mul�
&batch_normalization_52/batchnorm/mul_1Muldense_43/MatMul:product:0(batch_normalization_52/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_52/batchnorm/mul_1�
&batch_normalization_52/batchnorm/mul_2Mul/batch_normalization_52/moments/Squeeze:output:0(batch_normalization_52/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_52/batchnorm/mul_2�
$batch_normalization_52/batchnorm/subSub2batch_normalization_52/Cast/ReadVariableOp:value:0*batch_normalization_52/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_52/batchnorm/sub�
&batch_normalization_52/batchnorm/add_1AddV2*batch_normalization_52/batchnorm/mul_1:z:0(batch_normalization_52/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_52/batchnorm/add_1v
Relu_3Relu*batch_normalization_52/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_44/MatMul/ReadVariableOp�
dense_44/MatMulMatMulRelu_3:activations:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_44/MatMul�
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_44/BiasAdd/ReadVariableOp�
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_44/BiasAddt
IdentityIdentitydense_44/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp'^batch_normalization_48/AssignMovingAvg6^batch_normalization_48/AssignMovingAvg/ReadVariableOp)^batch_normalization_48/AssignMovingAvg_18^batch_normalization_48/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_48/Cast/ReadVariableOp-^batch_normalization_48/Cast_1/ReadVariableOp'^batch_normalization_49/AssignMovingAvg6^batch_normalization_49/AssignMovingAvg/ReadVariableOp)^batch_normalization_49/AssignMovingAvg_18^batch_normalization_49/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_49/Cast/ReadVariableOp-^batch_normalization_49/Cast_1/ReadVariableOp'^batch_normalization_50/AssignMovingAvg6^batch_normalization_50/AssignMovingAvg/ReadVariableOp)^batch_normalization_50/AssignMovingAvg_18^batch_normalization_50/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_50/Cast/ReadVariableOp-^batch_normalization_50/Cast_1/ReadVariableOp'^batch_normalization_51/AssignMovingAvg6^batch_normalization_51/AssignMovingAvg/ReadVariableOp)^batch_normalization_51/AssignMovingAvg_18^batch_normalization_51/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_51/Cast/ReadVariableOp-^batch_normalization_51/Cast_1/ReadVariableOp'^batch_normalization_52/AssignMovingAvg6^batch_normalization_52/AssignMovingAvg/ReadVariableOp)^batch_normalization_52/AssignMovingAvg_18^batch_normalization_52/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_52/Cast/ReadVariableOp-^batch_normalization_52/Cast_1/ReadVariableOp^dense_40/MatMul/ReadVariableOp^dense_41/MatMul/ReadVariableOp^dense_42/MatMul/ReadVariableOp^dense_43/MatMul/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_48/AssignMovingAvg&batch_normalization_48/AssignMovingAvg2n
5batch_normalization_48/AssignMovingAvg/ReadVariableOp5batch_normalization_48/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_48/AssignMovingAvg_1(batch_normalization_48/AssignMovingAvg_12r
7batch_normalization_48/AssignMovingAvg_1/ReadVariableOp7batch_normalization_48/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_48/Cast/ReadVariableOp*batch_normalization_48/Cast/ReadVariableOp2\
,batch_normalization_48/Cast_1/ReadVariableOp,batch_normalization_48/Cast_1/ReadVariableOp2P
&batch_normalization_49/AssignMovingAvg&batch_normalization_49/AssignMovingAvg2n
5batch_normalization_49/AssignMovingAvg/ReadVariableOp5batch_normalization_49/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_49/AssignMovingAvg_1(batch_normalization_49/AssignMovingAvg_12r
7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_49/Cast/ReadVariableOp*batch_normalization_49/Cast/ReadVariableOp2\
,batch_normalization_49/Cast_1/ReadVariableOp,batch_normalization_49/Cast_1/ReadVariableOp2P
&batch_normalization_50/AssignMovingAvg&batch_normalization_50/AssignMovingAvg2n
5batch_normalization_50/AssignMovingAvg/ReadVariableOp5batch_normalization_50/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_50/AssignMovingAvg_1(batch_normalization_50/AssignMovingAvg_12r
7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_50/Cast/ReadVariableOp*batch_normalization_50/Cast/ReadVariableOp2\
,batch_normalization_50/Cast_1/ReadVariableOp,batch_normalization_50/Cast_1/ReadVariableOp2P
&batch_normalization_51/AssignMovingAvg&batch_normalization_51/AssignMovingAvg2n
5batch_normalization_51/AssignMovingAvg/ReadVariableOp5batch_normalization_51/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_51/AssignMovingAvg_1(batch_normalization_51/AssignMovingAvg_12r
7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_51/Cast/ReadVariableOp*batch_normalization_51/Cast/ReadVariableOp2\
,batch_normalization_51/Cast_1/ReadVariableOp,batch_normalization_51/Cast_1/ReadVariableOp2P
&batch_normalization_52/AssignMovingAvg&batch_normalization_52/AssignMovingAvg2n
5batch_normalization_52/AssignMovingAvg/ReadVariableOp5batch_normalization_52/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_52/AssignMovingAvg_1(batch_normalization_52/AssignMovingAvg_12r
7batch_normalization_52/AssignMovingAvg_1/ReadVariableOp7batch_normalization_52/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_52/Cast/ReadVariableOp*batch_normalization_52/Cast/ReadVariableOp2\
,batch_normalization_52/Cast_1/ReadVariableOp,batch_normalization_52/Cast_1/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������


_user_specified_namex
�
}
)__inference_dense_41_layer_call_fn_330577

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
D__inference_dense_41_layer_call_and_return_conditional_losses_3287172
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
�E
�
__inference__traced_save_330725
file_prefixR
Nsavev2_feed_forward_sub_net_8_batch_normalization_48_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_8_batch_normalization_48_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_8_batch_normalization_49_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_8_batch_normalization_49_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_8_batch_normalization_50_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_8_batch_normalization_50_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_8_batch_normalization_51_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_8_batch_normalization_51_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_8_batch_normalization_52_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_8_batch_normalization_52_beta_read_readvariableopX
Tsavev2_feed_forward_sub_net_8_batch_normalization_48_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_8_batch_normalization_48_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_8_batch_normalization_49_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_8_batch_normalization_49_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_8_batch_normalization_50_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_8_batch_normalization_50_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_8_batch_normalization_51_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_8_batch_normalization_51_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_8_batch_normalization_52_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_8_batch_normalization_52_moving_variance_read_readvariableopE
Asavev2_feed_forward_sub_net_8_dense_40_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_8_dense_41_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_8_dense_42_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_8_dense_43_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_8_dense_44_kernel_read_readvariableopC
?savev2_feed_forward_sub_net_8_dense_44_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Nsavev2_feed_forward_sub_net_8_batch_normalization_48_gamma_read_readvariableopMsavev2_feed_forward_sub_net_8_batch_normalization_48_beta_read_readvariableopNsavev2_feed_forward_sub_net_8_batch_normalization_49_gamma_read_readvariableopMsavev2_feed_forward_sub_net_8_batch_normalization_49_beta_read_readvariableopNsavev2_feed_forward_sub_net_8_batch_normalization_50_gamma_read_readvariableopMsavev2_feed_forward_sub_net_8_batch_normalization_50_beta_read_readvariableopNsavev2_feed_forward_sub_net_8_batch_normalization_51_gamma_read_readvariableopMsavev2_feed_forward_sub_net_8_batch_normalization_51_beta_read_readvariableopNsavev2_feed_forward_sub_net_8_batch_normalization_52_gamma_read_readvariableopMsavev2_feed_forward_sub_net_8_batch_normalization_52_beta_read_readvariableopTsavev2_feed_forward_sub_net_8_batch_normalization_48_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_8_batch_normalization_48_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_8_batch_normalization_49_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_8_batch_normalization_49_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_8_batch_normalization_50_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_8_batch_normalization_50_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_8_batch_normalization_51_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_8_batch_normalization_51_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_8_batch_normalization_52_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_8_batch_normalization_52_moving_variance_read_readvariableopAsavev2_feed_forward_sub_net_8_dense_40_kernel_read_readvariableopAsavev2_feed_forward_sub_net_8_dense_41_kernel_read_readvariableopAsavev2_feed_forward_sub_net_8_dense_42_kernel_read_readvariableopAsavev2_feed_forward_sub_net_8_dense_43_kernel_read_readvariableopAsavev2_feed_forward_sub_net_8_dense_44_kernel_read_readvariableop?savev2_feed_forward_sub_net_8_dense_44_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�
�
D__inference_dense_42_layer_call_and_return_conditional_losses_330584

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
�+
�
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_330441

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
)__inference_dense_43_layer_call_fn_330605

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
D__inference_dense_43_layer_call_and_return_conditional_losses_3287592
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
7__inference_batch_normalization_50_layer_call_fn_330385

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
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_3282612
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
D__inference_dense_40_layer_call_and_return_conditional_losses_328696

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
7__inference_batch_normalization_51_layer_call_fn_330454

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
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_3283652
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
R__inference_batch_normalization_48_layer_call_and_return_conditional_losses_327929

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
�+
�
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_330523

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
7__inference_feed_forward_sub_net_8_layer_call_fn_330082
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
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_3290162
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
�+
�
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_328593

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
7__inference_batch_normalization_52_layer_call_fn_330536

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
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_3285312
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
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_330405

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
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329911
input_1L
>batch_normalization_48_assignmovingavg_readvariableop_resource:
N
@batch_normalization_48_assignmovingavg_1_readvariableop_resource:
A
3batch_normalization_48_cast_readvariableop_resource:
C
5batch_normalization_48_cast_1_readvariableop_resource:
9
'dense_40_matmul_readvariableop_resource:
L
>batch_normalization_49_assignmovingavg_readvariableop_resource:N
@batch_normalization_49_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_49_cast_readvariableop_resource:C
5batch_normalization_49_cast_1_readvariableop_resource:9
'dense_41_matmul_readvariableop_resource:L
>batch_normalization_50_assignmovingavg_readvariableop_resource:N
@batch_normalization_50_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_50_cast_readvariableop_resource:C
5batch_normalization_50_cast_1_readvariableop_resource:9
'dense_42_matmul_readvariableop_resource:L
>batch_normalization_51_assignmovingavg_readvariableop_resource:N
@batch_normalization_51_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_51_cast_readvariableop_resource:C
5batch_normalization_51_cast_1_readvariableop_resource:9
'dense_43_matmul_readvariableop_resource:L
>batch_normalization_52_assignmovingavg_readvariableop_resource:N
@batch_normalization_52_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_52_cast_readvariableop_resource:C
5batch_normalization_52_cast_1_readvariableop_resource:9
'dense_44_matmul_readvariableop_resource:
6
(dense_44_biasadd_readvariableop_resource:

identity��&batch_normalization_48/AssignMovingAvg�5batch_normalization_48/AssignMovingAvg/ReadVariableOp�(batch_normalization_48/AssignMovingAvg_1�7batch_normalization_48/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_48/Cast/ReadVariableOp�,batch_normalization_48/Cast_1/ReadVariableOp�&batch_normalization_49/AssignMovingAvg�5batch_normalization_49/AssignMovingAvg/ReadVariableOp�(batch_normalization_49/AssignMovingAvg_1�7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_49/Cast/ReadVariableOp�,batch_normalization_49/Cast_1/ReadVariableOp�&batch_normalization_50/AssignMovingAvg�5batch_normalization_50/AssignMovingAvg/ReadVariableOp�(batch_normalization_50/AssignMovingAvg_1�7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_50/Cast/ReadVariableOp�,batch_normalization_50/Cast_1/ReadVariableOp�&batch_normalization_51/AssignMovingAvg�5batch_normalization_51/AssignMovingAvg/ReadVariableOp�(batch_normalization_51/AssignMovingAvg_1�7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_51/Cast/ReadVariableOp�,batch_normalization_51/Cast_1/ReadVariableOp�&batch_normalization_52/AssignMovingAvg�5batch_normalization_52/AssignMovingAvg/ReadVariableOp�(batch_normalization_52/AssignMovingAvg_1�7batch_normalization_52/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_52/Cast/ReadVariableOp�,batch_normalization_52/Cast_1/ReadVariableOp�dense_40/MatMul/ReadVariableOp�dense_41/MatMul/ReadVariableOp�dense_42/MatMul/ReadVariableOp�dense_43/MatMul/ReadVariableOp�dense_44/BiasAdd/ReadVariableOp�dense_44/MatMul/ReadVariableOp�
5batch_normalization_48/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_48/moments/mean/reduction_indices�
#batch_normalization_48/moments/meanMeaninput_1>batch_normalization_48/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_48/moments/mean�
+batch_normalization_48/moments/StopGradientStopGradient,batch_normalization_48/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_48/moments/StopGradient�
0batch_normalization_48/moments/SquaredDifferenceSquaredDifferenceinput_14batch_normalization_48/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������
22
0batch_normalization_48/moments/SquaredDifference�
9batch_normalization_48/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_48/moments/variance/reduction_indices�
'batch_normalization_48/moments/varianceMean4batch_normalization_48/moments/SquaredDifference:z:0Bbatch_normalization_48/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_48/moments/variance�
&batch_normalization_48/moments/SqueezeSqueeze,batch_normalization_48/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_48/moments/Squeeze�
(batch_normalization_48/moments/Squeeze_1Squeeze0batch_normalization_48/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_48/moments/Squeeze_1�
,batch_normalization_48/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_48/AssignMovingAvg/decay�
+batch_normalization_48/AssignMovingAvg/CastCast5batch_normalization_48/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_48/AssignMovingAvg/Cast�
5batch_normalization_48/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_48_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_48/AssignMovingAvg/ReadVariableOp�
*batch_normalization_48/AssignMovingAvg/subSub=batch_normalization_48/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_48/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_48/AssignMovingAvg/sub�
*batch_normalization_48/AssignMovingAvg/mulMul.batch_normalization_48/AssignMovingAvg/sub:z:0/batch_normalization_48/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_48/AssignMovingAvg/mul�
&batch_normalization_48/AssignMovingAvgAssignSubVariableOp>batch_normalization_48_assignmovingavg_readvariableop_resource.batch_normalization_48/AssignMovingAvg/mul:z:06^batch_normalization_48/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_48/AssignMovingAvg�
.batch_normalization_48/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_48/AssignMovingAvg_1/decay�
-batch_normalization_48/AssignMovingAvg_1/CastCast7batch_normalization_48/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_48/AssignMovingAvg_1/Cast�
7batch_normalization_48/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_48_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_48/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_48/AssignMovingAvg_1/subSub?batch_normalization_48/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_48/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_48/AssignMovingAvg_1/sub�
,batch_normalization_48/AssignMovingAvg_1/mulMul0batch_normalization_48/AssignMovingAvg_1/sub:z:01batch_normalization_48/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_48/AssignMovingAvg_1/mul�
(batch_normalization_48/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_48_assignmovingavg_1_readvariableop_resource0batch_normalization_48/AssignMovingAvg_1/mul:z:08^batch_normalization_48/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_48/AssignMovingAvg_1�
*batch_normalization_48/Cast/ReadVariableOpReadVariableOp3batch_normalization_48_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_48/Cast/ReadVariableOp�
,batch_normalization_48/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_48_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_48/Cast_1/ReadVariableOp�
&batch_normalization_48/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_48/batchnorm/add/y�
$batch_normalization_48/batchnorm/addAddV21batch_normalization_48/moments/Squeeze_1:output:0/batch_normalization_48/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_48/batchnorm/add�
&batch_normalization_48/batchnorm/RsqrtRsqrt(batch_normalization_48/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_48/batchnorm/Rsqrt�
$batch_normalization_48/batchnorm/mulMul*batch_normalization_48/batchnorm/Rsqrt:y:04batch_normalization_48/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_48/batchnorm/mul�
&batch_normalization_48/batchnorm/mul_1Mulinput_1(batch_normalization_48/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_48/batchnorm/mul_1�
&batch_normalization_48/batchnorm/mul_2Mul/batch_normalization_48/moments/Squeeze:output:0(batch_normalization_48/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_48/batchnorm/mul_2�
$batch_normalization_48/batchnorm/subSub2batch_normalization_48/Cast/ReadVariableOp:value:0*batch_normalization_48/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_48/batchnorm/sub�
&batch_normalization_48/batchnorm/add_1AddV2*batch_normalization_48/batchnorm/mul_1:z:0(batch_normalization_48/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_48/batchnorm/add_1�
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_40/MatMul/ReadVariableOp�
dense_40/MatMulMatMul*batch_normalization_48/batchnorm/add_1:z:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_40/MatMul�
5batch_normalization_49/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_49/moments/mean/reduction_indices�
#batch_normalization_49/moments/meanMeandense_40/MatMul:product:0>batch_normalization_49/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_49/moments/mean�
+batch_normalization_49/moments/StopGradientStopGradient,batch_normalization_49/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_49/moments/StopGradient�
0batch_normalization_49/moments/SquaredDifferenceSquaredDifferencedense_40/MatMul:product:04batch_normalization_49/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_49/moments/SquaredDifference�
9batch_normalization_49/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_49/moments/variance/reduction_indices�
'batch_normalization_49/moments/varianceMean4batch_normalization_49/moments/SquaredDifference:z:0Bbatch_normalization_49/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_49/moments/variance�
&batch_normalization_49/moments/SqueezeSqueeze,batch_normalization_49/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_49/moments/Squeeze�
(batch_normalization_49/moments/Squeeze_1Squeeze0batch_normalization_49/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_49/moments/Squeeze_1�
,batch_normalization_49/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_49/AssignMovingAvg/decay�
+batch_normalization_49/AssignMovingAvg/CastCast5batch_normalization_49/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_49/AssignMovingAvg/Cast�
5batch_normalization_49/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_49_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_49/AssignMovingAvg/ReadVariableOp�
*batch_normalization_49/AssignMovingAvg/subSub=batch_normalization_49/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_49/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_49/AssignMovingAvg/sub�
*batch_normalization_49/AssignMovingAvg/mulMul.batch_normalization_49/AssignMovingAvg/sub:z:0/batch_normalization_49/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_49/AssignMovingAvg/mul�
&batch_normalization_49/AssignMovingAvgAssignSubVariableOp>batch_normalization_49_assignmovingavg_readvariableop_resource.batch_normalization_49/AssignMovingAvg/mul:z:06^batch_normalization_49/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_49/AssignMovingAvg�
.batch_normalization_49/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_49/AssignMovingAvg_1/decay�
-batch_normalization_49/AssignMovingAvg_1/CastCast7batch_normalization_49/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_49/AssignMovingAvg_1/Cast�
7batch_normalization_49/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_49_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_49/AssignMovingAvg_1/subSub?batch_normalization_49/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_49/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_49/AssignMovingAvg_1/sub�
,batch_normalization_49/AssignMovingAvg_1/mulMul0batch_normalization_49/AssignMovingAvg_1/sub:z:01batch_normalization_49/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_49/AssignMovingAvg_1/mul�
(batch_normalization_49/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_49_assignmovingavg_1_readvariableop_resource0batch_normalization_49/AssignMovingAvg_1/mul:z:08^batch_normalization_49/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_49/AssignMovingAvg_1�
*batch_normalization_49/Cast/ReadVariableOpReadVariableOp3batch_normalization_49_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_49/Cast/ReadVariableOp�
,batch_normalization_49/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_49_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_49/Cast_1/ReadVariableOp�
&batch_normalization_49/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_49/batchnorm/add/y�
$batch_normalization_49/batchnorm/addAddV21batch_normalization_49/moments/Squeeze_1:output:0/batch_normalization_49/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/add�
&batch_normalization_49/batchnorm/RsqrtRsqrt(batch_normalization_49/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_49/batchnorm/Rsqrt�
$batch_normalization_49/batchnorm/mulMul*batch_normalization_49/batchnorm/Rsqrt:y:04batch_normalization_49/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/mul�
&batch_normalization_49/batchnorm/mul_1Muldense_40/MatMul:product:0(batch_normalization_49/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_49/batchnorm/mul_1�
&batch_normalization_49/batchnorm/mul_2Mul/batch_normalization_49/moments/Squeeze:output:0(batch_normalization_49/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_49/batchnorm/mul_2�
$batch_normalization_49/batchnorm/subSub2batch_normalization_49/Cast/ReadVariableOp:value:0*batch_normalization_49/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/sub�
&batch_normalization_49/batchnorm/add_1AddV2*batch_normalization_49/batchnorm/mul_1:z:0(batch_normalization_49/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_49/batchnorm/add_1r
ReluRelu*batch_normalization_49/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_41/MatMul/ReadVariableOp�
dense_41/MatMulMatMulRelu:activations:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_41/MatMul�
5batch_normalization_50/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_50/moments/mean/reduction_indices�
#batch_normalization_50/moments/meanMeandense_41/MatMul:product:0>batch_normalization_50/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_50/moments/mean�
+batch_normalization_50/moments/StopGradientStopGradient,batch_normalization_50/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_50/moments/StopGradient�
0batch_normalization_50/moments/SquaredDifferenceSquaredDifferencedense_41/MatMul:product:04batch_normalization_50/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_50/moments/SquaredDifference�
9batch_normalization_50/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_50/moments/variance/reduction_indices�
'batch_normalization_50/moments/varianceMean4batch_normalization_50/moments/SquaredDifference:z:0Bbatch_normalization_50/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_50/moments/variance�
&batch_normalization_50/moments/SqueezeSqueeze,batch_normalization_50/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_50/moments/Squeeze�
(batch_normalization_50/moments/Squeeze_1Squeeze0batch_normalization_50/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_50/moments/Squeeze_1�
,batch_normalization_50/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_50/AssignMovingAvg/decay�
+batch_normalization_50/AssignMovingAvg/CastCast5batch_normalization_50/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_50/AssignMovingAvg/Cast�
5batch_normalization_50/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_50_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_50/AssignMovingAvg/ReadVariableOp�
*batch_normalization_50/AssignMovingAvg/subSub=batch_normalization_50/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_50/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_50/AssignMovingAvg/sub�
*batch_normalization_50/AssignMovingAvg/mulMul.batch_normalization_50/AssignMovingAvg/sub:z:0/batch_normalization_50/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_50/AssignMovingAvg/mul�
&batch_normalization_50/AssignMovingAvgAssignSubVariableOp>batch_normalization_50_assignmovingavg_readvariableop_resource.batch_normalization_50/AssignMovingAvg/mul:z:06^batch_normalization_50/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_50/AssignMovingAvg�
.batch_normalization_50/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_50/AssignMovingAvg_1/decay�
-batch_normalization_50/AssignMovingAvg_1/CastCast7batch_normalization_50/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_50/AssignMovingAvg_1/Cast�
7batch_normalization_50/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_50_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_50/AssignMovingAvg_1/subSub?batch_normalization_50/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_50/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_50/AssignMovingAvg_1/sub�
,batch_normalization_50/AssignMovingAvg_1/mulMul0batch_normalization_50/AssignMovingAvg_1/sub:z:01batch_normalization_50/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_50/AssignMovingAvg_1/mul�
(batch_normalization_50/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_50_assignmovingavg_1_readvariableop_resource0batch_normalization_50/AssignMovingAvg_1/mul:z:08^batch_normalization_50/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_50/AssignMovingAvg_1�
*batch_normalization_50/Cast/ReadVariableOpReadVariableOp3batch_normalization_50_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_50/Cast/ReadVariableOp�
,batch_normalization_50/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_50_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_50/Cast_1/ReadVariableOp�
&batch_normalization_50/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_50/batchnorm/add/y�
$batch_normalization_50/batchnorm/addAddV21batch_normalization_50/moments/Squeeze_1:output:0/batch_normalization_50/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/add�
&batch_normalization_50/batchnorm/RsqrtRsqrt(batch_normalization_50/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_50/batchnorm/Rsqrt�
$batch_normalization_50/batchnorm/mulMul*batch_normalization_50/batchnorm/Rsqrt:y:04batch_normalization_50/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/mul�
&batch_normalization_50/batchnorm/mul_1Muldense_41/MatMul:product:0(batch_normalization_50/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_50/batchnorm/mul_1�
&batch_normalization_50/batchnorm/mul_2Mul/batch_normalization_50/moments/Squeeze:output:0(batch_normalization_50/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_50/batchnorm/mul_2�
$batch_normalization_50/batchnorm/subSub2batch_normalization_50/Cast/ReadVariableOp:value:0*batch_normalization_50/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/sub�
&batch_normalization_50/batchnorm/add_1AddV2*batch_normalization_50/batchnorm/mul_1:z:0(batch_normalization_50/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_50/batchnorm/add_1v
Relu_1Relu*batch_normalization_50/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_42/MatMul/ReadVariableOp�
dense_42/MatMulMatMulRelu_1:activations:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_42/MatMul�
5batch_normalization_51/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_51/moments/mean/reduction_indices�
#batch_normalization_51/moments/meanMeandense_42/MatMul:product:0>batch_normalization_51/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_51/moments/mean�
+batch_normalization_51/moments/StopGradientStopGradient,batch_normalization_51/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_51/moments/StopGradient�
0batch_normalization_51/moments/SquaredDifferenceSquaredDifferencedense_42/MatMul:product:04batch_normalization_51/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_51/moments/SquaredDifference�
9batch_normalization_51/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_51/moments/variance/reduction_indices�
'batch_normalization_51/moments/varianceMean4batch_normalization_51/moments/SquaredDifference:z:0Bbatch_normalization_51/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_51/moments/variance�
&batch_normalization_51/moments/SqueezeSqueeze,batch_normalization_51/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_51/moments/Squeeze�
(batch_normalization_51/moments/Squeeze_1Squeeze0batch_normalization_51/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_51/moments/Squeeze_1�
,batch_normalization_51/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_51/AssignMovingAvg/decay�
+batch_normalization_51/AssignMovingAvg/CastCast5batch_normalization_51/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_51/AssignMovingAvg/Cast�
5batch_normalization_51/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_51_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_51/AssignMovingAvg/ReadVariableOp�
*batch_normalization_51/AssignMovingAvg/subSub=batch_normalization_51/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_51/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_51/AssignMovingAvg/sub�
*batch_normalization_51/AssignMovingAvg/mulMul.batch_normalization_51/AssignMovingAvg/sub:z:0/batch_normalization_51/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_51/AssignMovingAvg/mul�
&batch_normalization_51/AssignMovingAvgAssignSubVariableOp>batch_normalization_51_assignmovingavg_readvariableop_resource.batch_normalization_51/AssignMovingAvg/mul:z:06^batch_normalization_51/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_51/AssignMovingAvg�
.batch_normalization_51/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_51/AssignMovingAvg_1/decay�
-batch_normalization_51/AssignMovingAvg_1/CastCast7batch_normalization_51/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_51/AssignMovingAvg_1/Cast�
7batch_normalization_51/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_51_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_51/AssignMovingAvg_1/subSub?batch_normalization_51/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_51/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_51/AssignMovingAvg_1/sub�
,batch_normalization_51/AssignMovingAvg_1/mulMul0batch_normalization_51/AssignMovingAvg_1/sub:z:01batch_normalization_51/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_51/AssignMovingAvg_1/mul�
(batch_normalization_51/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_51_assignmovingavg_1_readvariableop_resource0batch_normalization_51/AssignMovingAvg_1/mul:z:08^batch_normalization_51/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_51/AssignMovingAvg_1�
*batch_normalization_51/Cast/ReadVariableOpReadVariableOp3batch_normalization_51_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_51/Cast/ReadVariableOp�
,batch_normalization_51/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_51_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_51/Cast_1/ReadVariableOp�
&batch_normalization_51/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_51/batchnorm/add/y�
$batch_normalization_51/batchnorm/addAddV21batch_normalization_51/moments/Squeeze_1:output:0/batch_normalization_51/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/add�
&batch_normalization_51/batchnorm/RsqrtRsqrt(batch_normalization_51/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_51/batchnorm/Rsqrt�
$batch_normalization_51/batchnorm/mulMul*batch_normalization_51/batchnorm/Rsqrt:y:04batch_normalization_51/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/mul�
&batch_normalization_51/batchnorm/mul_1Muldense_42/MatMul:product:0(batch_normalization_51/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_51/batchnorm/mul_1�
&batch_normalization_51/batchnorm/mul_2Mul/batch_normalization_51/moments/Squeeze:output:0(batch_normalization_51/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_51/batchnorm/mul_2�
$batch_normalization_51/batchnorm/subSub2batch_normalization_51/Cast/ReadVariableOp:value:0*batch_normalization_51/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/sub�
&batch_normalization_51/batchnorm/add_1AddV2*batch_normalization_51/batchnorm/mul_1:z:0(batch_normalization_51/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_51/batchnorm/add_1v
Relu_2Relu*batch_normalization_51/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_43/MatMul/ReadVariableOp�
dense_43/MatMulMatMulRelu_2:activations:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_43/MatMul�
5batch_normalization_52/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_52/moments/mean/reduction_indices�
#batch_normalization_52/moments/meanMeandense_43/MatMul:product:0>batch_normalization_52/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_52/moments/mean�
+batch_normalization_52/moments/StopGradientStopGradient,batch_normalization_52/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_52/moments/StopGradient�
0batch_normalization_52/moments/SquaredDifferenceSquaredDifferencedense_43/MatMul:product:04batch_normalization_52/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_52/moments/SquaredDifference�
9batch_normalization_52/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_52/moments/variance/reduction_indices�
'batch_normalization_52/moments/varianceMean4batch_normalization_52/moments/SquaredDifference:z:0Bbatch_normalization_52/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_52/moments/variance�
&batch_normalization_52/moments/SqueezeSqueeze,batch_normalization_52/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_52/moments/Squeeze�
(batch_normalization_52/moments/Squeeze_1Squeeze0batch_normalization_52/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_52/moments/Squeeze_1�
,batch_normalization_52/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_52/AssignMovingAvg/decay�
+batch_normalization_52/AssignMovingAvg/CastCast5batch_normalization_52/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_52/AssignMovingAvg/Cast�
5batch_normalization_52/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_52_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_52/AssignMovingAvg/ReadVariableOp�
*batch_normalization_52/AssignMovingAvg/subSub=batch_normalization_52/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_52/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_52/AssignMovingAvg/sub�
*batch_normalization_52/AssignMovingAvg/mulMul.batch_normalization_52/AssignMovingAvg/sub:z:0/batch_normalization_52/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_52/AssignMovingAvg/mul�
&batch_normalization_52/AssignMovingAvgAssignSubVariableOp>batch_normalization_52_assignmovingavg_readvariableop_resource.batch_normalization_52/AssignMovingAvg/mul:z:06^batch_normalization_52/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_52/AssignMovingAvg�
.batch_normalization_52/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_52/AssignMovingAvg_1/decay�
-batch_normalization_52/AssignMovingAvg_1/CastCast7batch_normalization_52/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_52/AssignMovingAvg_1/Cast�
7batch_normalization_52/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_52_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_52/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_52/AssignMovingAvg_1/subSub?batch_normalization_52/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_52/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_52/AssignMovingAvg_1/sub�
,batch_normalization_52/AssignMovingAvg_1/mulMul0batch_normalization_52/AssignMovingAvg_1/sub:z:01batch_normalization_52/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_52/AssignMovingAvg_1/mul�
(batch_normalization_52/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_52_assignmovingavg_1_readvariableop_resource0batch_normalization_52/AssignMovingAvg_1/mul:z:08^batch_normalization_52/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_52/AssignMovingAvg_1�
*batch_normalization_52/Cast/ReadVariableOpReadVariableOp3batch_normalization_52_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_52/Cast/ReadVariableOp�
,batch_normalization_52/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_52_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_52/Cast_1/ReadVariableOp�
&batch_normalization_52/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_52/batchnorm/add/y�
$batch_normalization_52/batchnorm/addAddV21batch_normalization_52/moments/Squeeze_1:output:0/batch_normalization_52/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_52/batchnorm/add�
&batch_normalization_52/batchnorm/RsqrtRsqrt(batch_normalization_52/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_52/batchnorm/Rsqrt�
$batch_normalization_52/batchnorm/mulMul*batch_normalization_52/batchnorm/Rsqrt:y:04batch_normalization_52/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_52/batchnorm/mul�
&batch_normalization_52/batchnorm/mul_1Muldense_43/MatMul:product:0(batch_normalization_52/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_52/batchnorm/mul_1�
&batch_normalization_52/batchnorm/mul_2Mul/batch_normalization_52/moments/Squeeze:output:0(batch_normalization_52/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_52/batchnorm/mul_2�
$batch_normalization_52/batchnorm/subSub2batch_normalization_52/Cast/ReadVariableOp:value:0*batch_normalization_52/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_52/batchnorm/sub�
&batch_normalization_52/batchnorm/add_1AddV2*batch_normalization_52/batchnorm/mul_1:z:0(batch_normalization_52/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_52/batchnorm/add_1v
Relu_3Relu*batch_normalization_52/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_44/MatMul/ReadVariableOp�
dense_44/MatMulMatMulRelu_3:activations:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_44/MatMul�
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_44/BiasAdd/ReadVariableOp�
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_44/BiasAddt
IdentityIdentitydense_44/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp'^batch_normalization_48/AssignMovingAvg6^batch_normalization_48/AssignMovingAvg/ReadVariableOp)^batch_normalization_48/AssignMovingAvg_18^batch_normalization_48/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_48/Cast/ReadVariableOp-^batch_normalization_48/Cast_1/ReadVariableOp'^batch_normalization_49/AssignMovingAvg6^batch_normalization_49/AssignMovingAvg/ReadVariableOp)^batch_normalization_49/AssignMovingAvg_18^batch_normalization_49/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_49/Cast/ReadVariableOp-^batch_normalization_49/Cast_1/ReadVariableOp'^batch_normalization_50/AssignMovingAvg6^batch_normalization_50/AssignMovingAvg/ReadVariableOp)^batch_normalization_50/AssignMovingAvg_18^batch_normalization_50/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_50/Cast/ReadVariableOp-^batch_normalization_50/Cast_1/ReadVariableOp'^batch_normalization_51/AssignMovingAvg6^batch_normalization_51/AssignMovingAvg/ReadVariableOp)^batch_normalization_51/AssignMovingAvg_18^batch_normalization_51/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_51/Cast/ReadVariableOp-^batch_normalization_51/Cast_1/ReadVariableOp'^batch_normalization_52/AssignMovingAvg6^batch_normalization_52/AssignMovingAvg/ReadVariableOp)^batch_normalization_52/AssignMovingAvg_18^batch_normalization_52/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_52/Cast/ReadVariableOp-^batch_normalization_52/Cast_1/ReadVariableOp^dense_40/MatMul/ReadVariableOp^dense_41/MatMul/ReadVariableOp^dense_42/MatMul/ReadVariableOp^dense_43/MatMul/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_48/AssignMovingAvg&batch_normalization_48/AssignMovingAvg2n
5batch_normalization_48/AssignMovingAvg/ReadVariableOp5batch_normalization_48/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_48/AssignMovingAvg_1(batch_normalization_48/AssignMovingAvg_12r
7batch_normalization_48/AssignMovingAvg_1/ReadVariableOp7batch_normalization_48/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_48/Cast/ReadVariableOp*batch_normalization_48/Cast/ReadVariableOp2\
,batch_normalization_48/Cast_1/ReadVariableOp,batch_normalization_48/Cast_1/ReadVariableOp2P
&batch_normalization_49/AssignMovingAvg&batch_normalization_49/AssignMovingAvg2n
5batch_normalization_49/AssignMovingAvg/ReadVariableOp5batch_normalization_49/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_49/AssignMovingAvg_1(batch_normalization_49/AssignMovingAvg_12r
7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp7batch_normalization_49/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_49/Cast/ReadVariableOp*batch_normalization_49/Cast/ReadVariableOp2\
,batch_normalization_49/Cast_1/ReadVariableOp,batch_normalization_49/Cast_1/ReadVariableOp2P
&batch_normalization_50/AssignMovingAvg&batch_normalization_50/AssignMovingAvg2n
5batch_normalization_50/AssignMovingAvg/ReadVariableOp5batch_normalization_50/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_50/AssignMovingAvg_1(batch_normalization_50/AssignMovingAvg_12r
7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp7batch_normalization_50/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_50/Cast/ReadVariableOp*batch_normalization_50/Cast/ReadVariableOp2\
,batch_normalization_50/Cast_1/ReadVariableOp,batch_normalization_50/Cast_1/ReadVariableOp2P
&batch_normalization_51/AssignMovingAvg&batch_normalization_51/AssignMovingAvg2n
5batch_normalization_51/AssignMovingAvg/ReadVariableOp5batch_normalization_51/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_51/AssignMovingAvg_1(batch_normalization_51/AssignMovingAvg_12r
7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp7batch_normalization_51/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_51/Cast/ReadVariableOp*batch_normalization_51/Cast/ReadVariableOp2\
,batch_normalization_51/Cast_1/ReadVariableOp,batch_normalization_51/Cast_1/ReadVariableOp2P
&batch_normalization_52/AssignMovingAvg&batch_normalization_52/AssignMovingAvg2n
5batch_normalization_52/AssignMovingAvg/ReadVariableOp5batch_normalization_52/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_52/AssignMovingAvg_1(batch_normalization_52/AssignMovingAvg_12r
7batch_normalization_52/AssignMovingAvg_1/ReadVariableOp7batch_normalization_52/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_52/Cast/ReadVariableOp*batch_normalization_52/Cast/ReadVariableOp2\
,batch_normalization_52/Cast_1/ReadVariableOp,batch_normalization_52/Cast_1/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
�+
�
R__inference_batch_normalization_49_layer_call_and_return_conditional_losses_330277

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
D__inference_dense_40_layer_call_and_return_conditional_losses_330556

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
�
�
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_328365

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
R__inference_batch_normalization_48_layer_call_and_return_conditional_losses_330159

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
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_330323

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
R__inference_batch_normalization_49_layer_call_and_return_conditional_losses_330241

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
R__inference_batch_normalization_48_layer_call_and_return_conditional_losses_327867

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
��
� 
!__inference__wrapped_model_327843
input_1X
Jfeed_forward_sub_net_8_batch_normalization_48_cast_readvariableop_resource:
Z
Lfeed_forward_sub_net_8_batch_normalization_48_cast_1_readvariableop_resource:
Z
Lfeed_forward_sub_net_8_batch_normalization_48_cast_2_readvariableop_resource:
Z
Lfeed_forward_sub_net_8_batch_normalization_48_cast_3_readvariableop_resource:
P
>feed_forward_sub_net_8_dense_40_matmul_readvariableop_resource:
X
Jfeed_forward_sub_net_8_batch_normalization_49_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_8_batch_normalization_49_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_8_batch_normalization_49_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_8_batch_normalization_49_cast_3_readvariableop_resource:P
>feed_forward_sub_net_8_dense_41_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_8_batch_normalization_50_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_8_batch_normalization_50_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_8_batch_normalization_50_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_8_batch_normalization_50_cast_3_readvariableop_resource:P
>feed_forward_sub_net_8_dense_42_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_8_batch_normalization_51_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_8_batch_normalization_51_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_8_batch_normalization_51_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_8_batch_normalization_51_cast_3_readvariableop_resource:P
>feed_forward_sub_net_8_dense_43_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_8_batch_normalization_52_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_8_batch_normalization_52_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_8_batch_normalization_52_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_8_batch_normalization_52_cast_3_readvariableop_resource:P
>feed_forward_sub_net_8_dense_44_matmul_readvariableop_resource:
M
?feed_forward_sub_net_8_dense_44_biasadd_readvariableop_resource:

identity��Afeed_forward_sub_net_8/batch_normalization_48/Cast/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_48/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_48/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_48/Cast_3/ReadVariableOp�Afeed_forward_sub_net_8/batch_normalization_49/Cast/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_49/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_49/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_49/Cast_3/ReadVariableOp�Afeed_forward_sub_net_8/batch_normalization_50/Cast/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_50/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_50/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_50/Cast_3/ReadVariableOp�Afeed_forward_sub_net_8/batch_normalization_51/Cast/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_51/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_51/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_51/Cast_3/ReadVariableOp�Afeed_forward_sub_net_8/batch_normalization_52/Cast/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_52/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_52/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_8/batch_normalization_52/Cast_3/ReadVariableOp�5feed_forward_sub_net_8/dense_40/MatMul/ReadVariableOp�5feed_forward_sub_net_8/dense_41/MatMul/ReadVariableOp�5feed_forward_sub_net_8/dense_42/MatMul/ReadVariableOp�5feed_forward_sub_net_8/dense_43/MatMul/ReadVariableOp�6feed_forward_sub_net_8/dense_44/BiasAdd/ReadVariableOp�5feed_forward_sub_net_8/dense_44/MatMul/ReadVariableOp�
Afeed_forward_sub_net_8/batch_normalization_48/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_8_batch_normalization_48_cast_readvariableop_resource*
_output_shapes
:
*
dtype02C
Afeed_forward_sub_net_8/batch_normalization_48/Cast/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_48/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_48_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_48/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_48/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_48_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_48/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_48/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_48_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_48/Cast_3/ReadVariableOp�
=feed_forward_sub_net_8/batch_normalization_48/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_8/batch_normalization_48/batchnorm/add/y�
;feed_forward_sub_net_8/batch_normalization_48/batchnorm/addAddV2Kfeed_forward_sub_net_8/batch_normalization_48/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_8/batch_normalization_48/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_8/batch_normalization_48/batchnorm/add�
=feed_forward_sub_net_8/batch_normalization_48/batchnorm/RsqrtRsqrt?feed_forward_sub_net_8/batch_normalization_48/batchnorm/add:z:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_8/batch_normalization_48/batchnorm/Rsqrt�
;feed_forward_sub_net_8/batch_normalization_48/batchnorm/mulMulAfeed_forward_sub_net_8/batch_normalization_48/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_8/batch_normalization_48/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_8/batch_normalization_48/batchnorm/mul�
=feed_forward_sub_net_8/batch_normalization_48/batchnorm/mul_1Mulinput_1?feed_forward_sub_net_8/batch_normalization_48/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2?
=feed_forward_sub_net_8/batch_normalization_48/batchnorm/mul_1�
=feed_forward_sub_net_8/batch_normalization_48/batchnorm/mul_2MulIfeed_forward_sub_net_8/batch_normalization_48/Cast/ReadVariableOp:value:0?feed_forward_sub_net_8/batch_normalization_48/batchnorm/mul:z:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_8/batch_normalization_48/batchnorm/mul_2�
;feed_forward_sub_net_8/batch_normalization_48/batchnorm/subSubKfeed_forward_sub_net_8/batch_normalization_48/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_8/batch_normalization_48/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_8/batch_normalization_48/batchnorm/sub�
=feed_forward_sub_net_8/batch_normalization_48/batchnorm/add_1AddV2Afeed_forward_sub_net_8/batch_normalization_48/batchnorm/mul_1:z:0?feed_forward_sub_net_8/batch_normalization_48/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2?
=feed_forward_sub_net_8/batch_normalization_48/batchnorm/add_1�
5feed_forward_sub_net_8/dense_40/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_8_dense_40_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5feed_forward_sub_net_8/dense_40/MatMul/ReadVariableOp�
&feed_forward_sub_net_8/dense_40/MatMulMatMulAfeed_forward_sub_net_8/batch_normalization_48/batchnorm/add_1:z:0=feed_forward_sub_net_8/dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&feed_forward_sub_net_8/dense_40/MatMul�
Afeed_forward_sub_net_8/batch_normalization_49/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_8_batch_normalization_49_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_8/batch_normalization_49/Cast/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_49/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_49_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_49/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_49/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_49_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_49/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_49/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_49_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_49/Cast_3/ReadVariableOp�
=feed_forward_sub_net_8/batch_normalization_49/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_8/batch_normalization_49/batchnorm/add/y�
;feed_forward_sub_net_8/batch_normalization_49/batchnorm/addAddV2Kfeed_forward_sub_net_8/batch_normalization_49/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_8/batch_normalization_49/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_8/batch_normalization_49/batchnorm/add�
=feed_forward_sub_net_8/batch_normalization_49/batchnorm/RsqrtRsqrt?feed_forward_sub_net_8/batch_normalization_49/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_8/batch_normalization_49/batchnorm/Rsqrt�
;feed_forward_sub_net_8/batch_normalization_49/batchnorm/mulMulAfeed_forward_sub_net_8/batch_normalization_49/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_8/batch_normalization_49/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_8/batch_normalization_49/batchnorm/mul�
=feed_forward_sub_net_8/batch_normalization_49/batchnorm/mul_1Mul0feed_forward_sub_net_8/dense_40/MatMul:product:0?feed_forward_sub_net_8/batch_normalization_49/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_8/batch_normalization_49/batchnorm/mul_1�
=feed_forward_sub_net_8/batch_normalization_49/batchnorm/mul_2MulIfeed_forward_sub_net_8/batch_normalization_49/Cast/ReadVariableOp:value:0?feed_forward_sub_net_8/batch_normalization_49/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_8/batch_normalization_49/batchnorm/mul_2�
;feed_forward_sub_net_8/batch_normalization_49/batchnorm/subSubKfeed_forward_sub_net_8/batch_normalization_49/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_8/batch_normalization_49/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_8/batch_normalization_49/batchnorm/sub�
=feed_forward_sub_net_8/batch_normalization_49/batchnorm/add_1AddV2Afeed_forward_sub_net_8/batch_normalization_49/batchnorm/mul_1:z:0?feed_forward_sub_net_8/batch_normalization_49/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_8/batch_normalization_49/batchnorm/add_1�
feed_forward_sub_net_8/ReluReluAfeed_forward_sub_net_8/batch_normalization_49/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_8/Relu�
5feed_forward_sub_net_8/dense_41/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_8_dense_41_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_8/dense_41/MatMul/ReadVariableOp�
&feed_forward_sub_net_8/dense_41/MatMulMatMul)feed_forward_sub_net_8/Relu:activations:0=feed_forward_sub_net_8/dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&feed_forward_sub_net_8/dense_41/MatMul�
Afeed_forward_sub_net_8/batch_normalization_50/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_8_batch_normalization_50_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_8/batch_normalization_50/Cast/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_50/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_50_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_50/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_50/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_50_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_50/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_50/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_50_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_50/Cast_3/ReadVariableOp�
=feed_forward_sub_net_8/batch_normalization_50/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_8/batch_normalization_50/batchnorm/add/y�
;feed_forward_sub_net_8/batch_normalization_50/batchnorm/addAddV2Kfeed_forward_sub_net_8/batch_normalization_50/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_8/batch_normalization_50/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_8/batch_normalization_50/batchnorm/add�
=feed_forward_sub_net_8/batch_normalization_50/batchnorm/RsqrtRsqrt?feed_forward_sub_net_8/batch_normalization_50/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_8/batch_normalization_50/batchnorm/Rsqrt�
;feed_forward_sub_net_8/batch_normalization_50/batchnorm/mulMulAfeed_forward_sub_net_8/batch_normalization_50/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_8/batch_normalization_50/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_8/batch_normalization_50/batchnorm/mul�
=feed_forward_sub_net_8/batch_normalization_50/batchnorm/mul_1Mul0feed_forward_sub_net_8/dense_41/MatMul:product:0?feed_forward_sub_net_8/batch_normalization_50/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_8/batch_normalization_50/batchnorm/mul_1�
=feed_forward_sub_net_8/batch_normalization_50/batchnorm/mul_2MulIfeed_forward_sub_net_8/batch_normalization_50/Cast/ReadVariableOp:value:0?feed_forward_sub_net_8/batch_normalization_50/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_8/batch_normalization_50/batchnorm/mul_2�
;feed_forward_sub_net_8/batch_normalization_50/batchnorm/subSubKfeed_forward_sub_net_8/batch_normalization_50/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_8/batch_normalization_50/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_8/batch_normalization_50/batchnorm/sub�
=feed_forward_sub_net_8/batch_normalization_50/batchnorm/add_1AddV2Afeed_forward_sub_net_8/batch_normalization_50/batchnorm/mul_1:z:0?feed_forward_sub_net_8/batch_normalization_50/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_8/batch_normalization_50/batchnorm/add_1�
feed_forward_sub_net_8/Relu_1ReluAfeed_forward_sub_net_8/batch_normalization_50/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_8/Relu_1�
5feed_forward_sub_net_8/dense_42/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_8_dense_42_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_8/dense_42/MatMul/ReadVariableOp�
&feed_forward_sub_net_8/dense_42/MatMulMatMul+feed_forward_sub_net_8/Relu_1:activations:0=feed_forward_sub_net_8/dense_42/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&feed_forward_sub_net_8/dense_42/MatMul�
Afeed_forward_sub_net_8/batch_normalization_51/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_8_batch_normalization_51_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_8/batch_normalization_51/Cast/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_51/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_51_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_51/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_51/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_51_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_51/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_51/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_51_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_51/Cast_3/ReadVariableOp�
=feed_forward_sub_net_8/batch_normalization_51/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_8/batch_normalization_51/batchnorm/add/y�
;feed_forward_sub_net_8/batch_normalization_51/batchnorm/addAddV2Kfeed_forward_sub_net_8/batch_normalization_51/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_8/batch_normalization_51/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_8/batch_normalization_51/batchnorm/add�
=feed_forward_sub_net_8/batch_normalization_51/batchnorm/RsqrtRsqrt?feed_forward_sub_net_8/batch_normalization_51/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_8/batch_normalization_51/batchnorm/Rsqrt�
;feed_forward_sub_net_8/batch_normalization_51/batchnorm/mulMulAfeed_forward_sub_net_8/batch_normalization_51/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_8/batch_normalization_51/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_8/batch_normalization_51/batchnorm/mul�
=feed_forward_sub_net_8/batch_normalization_51/batchnorm/mul_1Mul0feed_forward_sub_net_8/dense_42/MatMul:product:0?feed_forward_sub_net_8/batch_normalization_51/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_8/batch_normalization_51/batchnorm/mul_1�
=feed_forward_sub_net_8/batch_normalization_51/batchnorm/mul_2MulIfeed_forward_sub_net_8/batch_normalization_51/Cast/ReadVariableOp:value:0?feed_forward_sub_net_8/batch_normalization_51/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_8/batch_normalization_51/batchnorm/mul_2�
;feed_forward_sub_net_8/batch_normalization_51/batchnorm/subSubKfeed_forward_sub_net_8/batch_normalization_51/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_8/batch_normalization_51/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_8/batch_normalization_51/batchnorm/sub�
=feed_forward_sub_net_8/batch_normalization_51/batchnorm/add_1AddV2Afeed_forward_sub_net_8/batch_normalization_51/batchnorm/mul_1:z:0?feed_forward_sub_net_8/batch_normalization_51/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_8/batch_normalization_51/batchnorm/add_1�
feed_forward_sub_net_8/Relu_2ReluAfeed_forward_sub_net_8/batch_normalization_51/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_8/Relu_2�
5feed_forward_sub_net_8/dense_43/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_8_dense_43_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_8/dense_43/MatMul/ReadVariableOp�
&feed_forward_sub_net_8/dense_43/MatMulMatMul+feed_forward_sub_net_8/Relu_2:activations:0=feed_forward_sub_net_8/dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&feed_forward_sub_net_8/dense_43/MatMul�
Afeed_forward_sub_net_8/batch_normalization_52/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_8_batch_normalization_52_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_8/batch_normalization_52/Cast/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_52/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_52_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_52/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_52/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_52_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_52/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_8/batch_normalization_52/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_8_batch_normalization_52_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_8/batch_normalization_52/Cast_3/ReadVariableOp�
=feed_forward_sub_net_8/batch_normalization_52/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_8/batch_normalization_52/batchnorm/add/y�
;feed_forward_sub_net_8/batch_normalization_52/batchnorm/addAddV2Kfeed_forward_sub_net_8/batch_normalization_52/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_8/batch_normalization_52/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_8/batch_normalization_52/batchnorm/add�
=feed_forward_sub_net_8/batch_normalization_52/batchnorm/RsqrtRsqrt?feed_forward_sub_net_8/batch_normalization_52/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_8/batch_normalization_52/batchnorm/Rsqrt�
;feed_forward_sub_net_8/batch_normalization_52/batchnorm/mulMulAfeed_forward_sub_net_8/batch_normalization_52/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_8/batch_normalization_52/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_8/batch_normalization_52/batchnorm/mul�
=feed_forward_sub_net_8/batch_normalization_52/batchnorm/mul_1Mul0feed_forward_sub_net_8/dense_43/MatMul:product:0?feed_forward_sub_net_8/batch_normalization_52/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_8/batch_normalization_52/batchnorm/mul_1�
=feed_forward_sub_net_8/batch_normalization_52/batchnorm/mul_2MulIfeed_forward_sub_net_8/batch_normalization_52/Cast/ReadVariableOp:value:0?feed_forward_sub_net_8/batch_normalization_52/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_8/batch_normalization_52/batchnorm/mul_2�
;feed_forward_sub_net_8/batch_normalization_52/batchnorm/subSubKfeed_forward_sub_net_8/batch_normalization_52/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_8/batch_normalization_52/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_8/batch_normalization_52/batchnorm/sub�
=feed_forward_sub_net_8/batch_normalization_52/batchnorm/add_1AddV2Afeed_forward_sub_net_8/batch_normalization_52/batchnorm/mul_1:z:0?feed_forward_sub_net_8/batch_normalization_52/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_8/batch_normalization_52/batchnorm/add_1�
feed_forward_sub_net_8/Relu_3ReluAfeed_forward_sub_net_8/batch_normalization_52/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_8/Relu_3�
5feed_forward_sub_net_8/dense_44/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_8_dense_44_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5feed_forward_sub_net_8/dense_44/MatMul/ReadVariableOp�
&feed_forward_sub_net_8/dense_44/MatMulMatMul+feed_forward_sub_net_8/Relu_3:activations:0=feed_forward_sub_net_8/dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2(
&feed_forward_sub_net_8/dense_44/MatMul�
6feed_forward_sub_net_8/dense_44/BiasAdd/ReadVariableOpReadVariableOp?feed_forward_sub_net_8_dense_44_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype028
6feed_forward_sub_net_8/dense_44/BiasAdd/ReadVariableOp�
'feed_forward_sub_net_8/dense_44/BiasAddBiasAdd0feed_forward_sub_net_8/dense_44/MatMul:product:0>feed_forward_sub_net_8/dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2)
'feed_forward_sub_net_8/dense_44/BiasAdd�
IdentityIdentity0feed_forward_sub_net_8/dense_44/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOpB^feed_forward_sub_net_8/batch_normalization_48/Cast/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_48/Cast_1/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_48/Cast_2/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_48/Cast_3/ReadVariableOpB^feed_forward_sub_net_8/batch_normalization_49/Cast/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_49/Cast_1/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_49/Cast_2/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_49/Cast_3/ReadVariableOpB^feed_forward_sub_net_8/batch_normalization_50/Cast/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_50/Cast_1/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_50/Cast_2/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_50/Cast_3/ReadVariableOpB^feed_forward_sub_net_8/batch_normalization_51/Cast/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_51/Cast_1/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_51/Cast_2/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_51/Cast_3/ReadVariableOpB^feed_forward_sub_net_8/batch_normalization_52/Cast/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_52/Cast_1/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_52/Cast_2/ReadVariableOpD^feed_forward_sub_net_8/batch_normalization_52/Cast_3/ReadVariableOp6^feed_forward_sub_net_8/dense_40/MatMul/ReadVariableOp6^feed_forward_sub_net_8/dense_41/MatMul/ReadVariableOp6^feed_forward_sub_net_8/dense_42/MatMul/ReadVariableOp6^feed_forward_sub_net_8/dense_43/MatMul/ReadVariableOp7^feed_forward_sub_net_8/dense_44/BiasAdd/ReadVariableOp6^feed_forward_sub_net_8/dense_44/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2�
Afeed_forward_sub_net_8/batch_normalization_48/Cast/ReadVariableOpAfeed_forward_sub_net_8/batch_normalization_48/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_48/Cast_1/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_48/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_48/Cast_2/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_48/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_48/Cast_3/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_48/Cast_3/ReadVariableOp2�
Afeed_forward_sub_net_8/batch_normalization_49/Cast/ReadVariableOpAfeed_forward_sub_net_8/batch_normalization_49/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_49/Cast_1/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_49/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_49/Cast_2/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_49/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_49/Cast_3/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_49/Cast_3/ReadVariableOp2�
Afeed_forward_sub_net_8/batch_normalization_50/Cast/ReadVariableOpAfeed_forward_sub_net_8/batch_normalization_50/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_50/Cast_1/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_50/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_50/Cast_2/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_50/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_50/Cast_3/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_50/Cast_3/ReadVariableOp2�
Afeed_forward_sub_net_8/batch_normalization_51/Cast/ReadVariableOpAfeed_forward_sub_net_8/batch_normalization_51/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_51/Cast_1/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_51/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_51/Cast_2/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_51/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_51/Cast_3/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_51/Cast_3/ReadVariableOp2�
Afeed_forward_sub_net_8/batch_normalization_52/Cast/ReadVariableOpAfeed_forward_sub_net_8/batch_normalization_52/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_52/Cast_1/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_52/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_52/Cast_2/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_52/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_8/batch_normalization_52/Cast_3/ReadVariableOpCfeed_forward_sub_net_8/batch_normalization_52/Cast_3/ReadVariableOp2n
5feed_forward_sub_net_8/dense_40/MatMul/ReadVariableOp5feed_forward_sub_net_8/dense_40/MatMul/ReadVariableOp2n
5feed_forward_sub_net_8/dense_41/MatMul/ReadVariableOp5feed_forward_sub_net_8/dense_41/MatMul/ReadVariableOp2n
5feed_forward_sub_net_8/dense_42/MatMul/ReadVariableOp5feed_forward_sub_net_8/dense_42/MatMul/ReadVariableOp2n
5feed_forward_sub_net_8/dense_43/MatMul/ReadVariableOp5feed_forward_sub_net_8/dense_43/MatMul/ReadVariableOp2p
6feed_forward_sub_net_8/dense_44/BiasAdd/ReadVariableOp6feed_forward_sub_net_8/dense_44/BiasAdd/ReadVariableOp2n
5feed_forward_sub_net_8/dense_44/MatMul/ReadVariableOp5feed_forward_sub_net_8/dense_44/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
�
�
7__inference_batch_normalization_51_layer_call_fn_330467

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
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_3284272
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
7__inference_batch_normalization_49_layer_call_fn_330303

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
R__inference_batch_normalization_49_layer_call_and_return_conditional_losses_3280952
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
)__inference_dense_42_layer_call_fn_330591

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
D__inference_dense_42_layer_call_and_return_conditional_losses_3287382
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
7__inference_feed_forward_sub_net_8_layer_call_fn_330139
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
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_3290162
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
7__inference_batch_normalization_52_layer_call_fn_330549

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
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_3285932
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
D__inference_dense_41_layer_call_and_return_conditional_losses_330570

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
�+
�
R__inference_batch_normalization_48_layer_call_and_return_conditional_losses_330195

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
��
�
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329725
input_1A
3batch_normalization_48_cast_readvariableop_resource:
C
5batch_normalization_48_cast_1_readvariableop_resource:
C
5batch_normalization_48_cast_2_readvariableop_resource:
C
5batch_normalization_48_cast_3_readvariableop_resource:
9
'dense_40_matmul_readvariableop_resource:
A
3batch_normalization_49_cast_readvariableop_resource:C
5batch_normalization_49_cast_1_readvariableop_resource:C
5batch_normalization_49_cast_2_readvariableop_resource:C
5batch_normalization_49_cast_3_readvariableop_resource:9
'dense_41_matmul_readvariableop_resource:A
3batch_normalization_50_cast_readvariableop_resource:C
5batch_normalization_50_cast_1_readvariableop_resource:C
5batch_normalization_50_cast_2_readvariableop_resource:C
5batch_normalization_50_cast_3_readvariableop_resource:9
'dense_42_matmul_readvariableop_resource:A
3batch_normalization_51_cast_readvariableop_resource:C
5batch_normalization_51_cast_1_readvariableop_resource:C
5batch_normalization_51_cast_2_readvariableop_resource:C
5batch_normalization_51_cast_3_readvariableop_resource:9
'dense_43_matmul_readvariableop_resource:A
3batch_normalization_52_cast_readvariableop_resource:C
5batch_normalization_52_cast_1_readvariableop_resource:C
5batch_normalization_52_cast_2_readvariableop_resource:C
5batch_normalization_52_cast_3_readvariableop_resource:9
'dense_44_matmul_readvariableop_resource:
6
(dense_44_biasadd_readvariableop_resource:

identity��*batch_normalization_48/Cast/ReadVariableOp�,batch_normalization_48/Cast_1/ReadVariableOp�,batch_normalization_48/Cast_2/ReadVariableOp�,batch_normalization_48/Cast_3/ReadVariableOp�*batch_normalization_49/Cast/ReadVariableOp�,batch_normalization_49/Cast_1/ReadVariableOp�,batch_normalization_49/Cast_2/ReadVariableOp�,batch_normalization_49/Cast_3/ReadVariableOp�*batch_normalization_50/Cast/ReadVariableOp�,batch_normalization_50/Cast_1/ReadVariableOp�,batch_normalization_50/Cast_2/ReadVariableOp�,batch_normalization_50/Cast_3/ReadVariableOp�*batch_normalization_51/Cast/ReadVariableOp�,batch_normalization_51/Cast_1/ReadVariableOp�,batch_normalization_51/Cast_2/ReadVariableOp�,batch_normalization_51/Cast_3/ReadVariableOp�*batch_normalization_52/Cast/ReadVariableOp�,batch_normalization_52/Cast_1/ReadVariableOp�,batch_normalization_52/Cast_2/ReadVariableOp�,batch_normalization_52/Cast_3/ReadVariableOp�dense_40/MatMul/ReadVariableOp�dense_41/MatMul/ReadVariableOp�dense_42/MatMul/ReadVariableOp�dense_43/MatMul/ReadVariableOp�dense_44/BiasAdd/ReadVariableOp�dense_44/MatMul/ReadVariableOp�
*batch_normalization_48/Cast/ReadVariableOpReadVariableOp3batch_normalization_48_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_48/Cast/ReadVariableOp�
,batch_normalization_48/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_48_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_48/Cast_1/ReadVariableOp�
,batch_normalization_48/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_48_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_48/Cast_2/ReadVariableOp�
,batch_normalization_48/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_48_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_48/Cast_3/ReadVariableOp�
&batch_normalization_48/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_48/batchnorm/add/y�
$batch_normalization_48/batchnorm/addAddV24batch_normalization_48/Cast_1/ReadVariableOp:value:0/batch_normalization_48/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_48/batchnorm/add�
&batch_normalization_48/batchnorm/RsqrtRsqrt(batch_normalization_48/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_48/batchnorm/Rsqrt�
$batch_normalization_48/batchnorm/mulMul*batch_normalization_48/batchnorm/Rsqrt:y:04batch_normalization_48/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_48/batchnorm/mul�
&batch_normalization_48/batchnorm/mul_1Mulinput_1(batch_normalization_48/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_48/batchnorm/mul_1�
&batch_normalization_48/batchnorm/mul_2Mul2batch_normalization_48/Cast/ReadVariableOp:value:0(batch_normalization_48/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_48/batchnorm/mul_2�
$batch_normalization_48/batchnorm/subSub4batch_normalization_48/Cast_2/ReadVariableOp:value:0*batch_normalization_48/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_48/batchnorm/sub�
&batch_normalization_48/batchnorm/add_1AddV2*batch_normalization_48/batchnorm/mul_1:z:0(batch_normalization_48/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_48/batchnorm/add_1�
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_40/MatMul/ReadVariableOp�
dense_40/MatMulMatMul*batch_normalization_48/batchnorm/add_1:z:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_40/MatMul�
*batch_normalization_49/Cast/ReadVariableOpReadVariableOp3batch_normalization_49_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_49/Cast/ReadVariableOp�
,batch_normalization_49/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_49_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_49/Cast_1/ReadVariableOp�
,batch_normalization_49/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_49_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_49/Cast_2/ReadVariableOp�
,batch_normalization_49/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_49_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_49/Cast_3/ReadVariableOp�
&batch_normalization_49/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_49/batchnorm/add/y�
$batch_normalization_49/batchnorm/addAddV24batch_normalization_49/Cast_1/ReadVariableOp:value:0/batch_normalization_49/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/add�
&batch_normalization_49/batchnorm/RsqrtRsqrt(batch_normalization_49/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_49/batchnorm/Rsqrt�
$batch_normalization_49/batchnorm/mulMul*batch_normalization_49/batchnorm/Rsqrt:y:04batch_normalization_49/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/mul�
&batch_normalization_49/batchnorm/mul_1Muldense_40/MatMul:product:0(batch_normalization_49/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_49/batchnorm/mul_1�
&batch_normalization_49/batchnorm/mul_2Mul2batch_normalization_49/Cast/ReadVariableOp:value:0(batch_normalization_49/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_49/batchnorm/mul_2�
$batch_normalization_49/batchnorm/subSub4batch_normalization_49/Cast_2/ReadVariableOp:value:0*batch_normalization_49/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_49/batchnorm/sub�
&batch_normalization_49/batchnorm/add_1AddV2*batch_normalization_49/batchnorm/mul_1:z:0(batch_normalization_49/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_49/batchnorm/add_1r
ReluRelu*batch_normalization_49/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_41/MatMul/ReadVariableOp�
dense_41/MatMulMatMulRelu:activations:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_41/MatMul�
*batch_normalization_50/Cast/ReadVariableOpReadVariableOp3batch_normalization_50_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_50/Cast/ReadVariableOp�
,batch_normalization_50/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_50_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_50/Cast_1/ReadVariableOp�
,batch_normalization_50/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_50_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_50/Cast_2/ReadVariableOp�
,batch_normalization_50/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_50_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_50/Cast_3/ReadVariableOp�
&batch_normalization_50/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_50/batchnorm/add/y�
$batch_normalization_50/batchnorm/addAddV24batch_normalization_50/Cast_1/ReadVariableOp:value:0/batch_normalization_50/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/add�
&batch_normalization_50/batchnorm/RsqrtRsqrt(batch_normalization_50/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_50/batchnorm/Rsqrt�
$batch_normalization_50/batchnorm/mulMul*batch_normalization_50/batchnorm/Rsqrt:y:04batch_normalization_50/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/mul�
&batch_normalization_50/batchnorm/mul_1Muldense_41/MatMul:product:0(batch_normalization_50/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_50/batchnorm/mul_1�
&batch_normalization_50/batchnorm/mul_2Mul2batch_normalization_50/Cast/ReadVariableOp:value:0(batch_normalization_50/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_50/batchnorm/mul_2�
$batch_normalization_50/batchnorm/subSub4batch_normalization_50/Cast_2/ReadVariableOp:value:0*batch_normalization_50/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_50/batchnorm/sub�
&batch_normalization_50/batchnorm/add_1AddV2*batch_normalization_50/batchnorm/mul_1:z:0(batch_normalization_50/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_50/batchnorm/add_1v
Relu_1Relu*batch_normalization_50/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_42/MatMul/ReadVariableOp�
dense_42/MatMulMatMulRelu_1:activations:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_42/MatMul�
*batch_normalization_51/Cast/ReadVariableOpReadVariableOp3batch_normalization_51_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_51/Cast/ReadVariableOp�
,batch_normalization_51/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_51_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_51/Cast_1/ReadVariableOp�
,batch_normalization_51/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_51_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_51/Cast_2/ReadVariableOp�
,batch_normalization_51/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_51_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_51/Cast_3/ReadVariableOp�
&batch_normalization_51/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_51/batchnorm/add/y�
$batch_normalization_51/batchnorm/addAddV24batch_normalization_51/Cast_1/ReadVariableOp:value:0/batch_normalization_51/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/add�
&batch_normalization_51/batchnorm/RsqrtRsqrt(batch_normalization_51/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_51/batchnorm/Rsqrt�
$batch_normalization_51/batchnorm/mulMul*batch_normalization_51/batchnorm/Rsqrt:y:04batch_normalization_51/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/mul�
&batch_normalization_51/batchnorm/mul_1Muldense_42/MatMul:product:0(batch_normalization_51/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_51/batchnorm/mul_1�
&batch_normalization_51/batchnorm/mul_2Mul2batch_normalization_51/Cast/ReadVariableOp:value:0(batch_normalization_51/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_51/batchnorm/mul_2�
$batch_normalization_51/batchnorm/subSub4batch_normalization_51/Cast_2/ReadVariableOp:value:0*batch_normalization_51/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_51/batchnorm/sub�
&batch_normalization_51/batchnorm/add_1AddV2*batch_normalization_51/batchnorm/mul_1:z:0(batch_normalization_51/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_51/batchnorm/add_1v
Relu_2Relu*batch_normalization_51/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_43/MatMul/ReadVariableOp�
dense_43/MatMulMatMulRelu_2:activations:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_43/MatMul�
*batch_normalization_52/Cast/ReadVariableOpReadVariableOp3batch_normalization_52_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_52/Cast/ReadVariableOp�
,batch_normalization_52/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_52_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_52/Cast_1/ReadVariableOp�
,batch_normalization_52/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_52_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_52/Cast_2/ReadVariableOp�
,batch_normalization_52/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_52_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_52/Cast_3/ReadVariableOp�
&batch_normalization_52/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_52/batchnorm/add/y�
$batch_normalization_52/batchnorm/addAddV24batch_normalization_52/Cast_1/ReadVariableOp:value:0/batch_normalization_52/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_52/batchnorm/add�
&batch_normalization_52/batchnorm/RsqrtRsqrt(batch_normalization_52/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_52/batchnorm/Rsqrt�
$batch_normalization_52/batchnorm/mulMul*batch_normalization_52/batchnorm/Rsqrt:y:04batch_normalization_52/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_52/batchnorm/mul�
&batch_normalization_52/batchnorm/mul_1Muldense_43/MatMul:product:0(batch_normalization_52/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_52/batchnorm/mul_1�
&batch_normalization_52/batchnorm/mul_2Mul2batch_normalization_52/Cast/ReadVariableOp:value:0(batch_normalization_52/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_52/batchnorm/mul_2�
$batch_normalization_52/batchnorm/subSub4batch_normalization_52/Cast_2/ReadVariableOp:value:0*batch_normalization_52/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_52/batchnorm/sub�
&batch_normalization_52/batchnorm/add_1AddV2*batch_normalization_52/batchnorm/mul_1:z:0(batch_normalization_52/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_52/batchnorm/add_1v
Relu_3Relu*batch_normalization_52/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_44/MatMul/ReadVariableOp�
dense_44/MatMulMatMulRelu_3:activations:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_44/MatMul�
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_44/BiasAdd/ReadVariableOp�
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_44/BiasAddt
IdentityIdentitydense_44/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�	
NoOpNoOp+^batch_normalization_48/Cast/ReadVariableOp-^batch_normalization_48/Cast_1/ReadVariableOp-^batch_normalization_48/Cast_2/ReadVariableOp-^batch_normalization_48/Cast_3/ReadVariableOp+^batch_normalization_49/Cast/ReadVariableOp-^batch_normalization_49/Cast_1/ReadVariableOp-^batch_normalization_49/Cast_2/ReadVariableOp-^batch_normalization_49/Cast_3/ReadVariableOp+^batch_normalization_50/Cast/ReadVariableOp-^batch_normalization_50/Cast_1/ReadVariableOp-^batch_normalization_50/Cast_2/ReadVariableOp-^batch_normalization_50/Cast_3/ReadVariableOp+^batch_normalization_51/Cast/ReadVariableOp-^batch_normalization_51/Cast_1/ReadVariableOp-^batch_normalization_51/Cast_2/ReadVariableOp-^batch_normalization_51/Cast_3/ReadVariableOp+^batch_normalization_52/Cast/ReadVariableOp-^batch_normalization_52/Cast_1/ReadVariableOp-^batch_normalization_52/Cast_2/ReadVariableOp-^batch_normalization_52/Cast_3/ReadVariableOp^dense_40/MatMul/ReadVariableOp^dense_41/MatMul/ReadVariableOp^dense_42/MatMul/ReadVariableOp^dense_43/MatMul/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_48/Cast/ReadVariableOp*batch_normalization_48/Cast/ReadVariableOp2\
,batch_normalization_48/Cast_1/ReadVariableOp,batch_normalization_48/Cast_1/ReadVariableOp2\
,batch_normalization_48/Cast_2/ReadVariableOp,batch_normalization_48/Cast_2/ReadVariableOp2\
,batch_normalization_48/Cast_3/ReadVariableOp,batch_normalization_48/Cast_3/ReadVariableOp2X
*batch_normalization_49/Cast/ReadVariableOp*batch_normalization_49/Cast/ReadVariableOp2\
,batch_normalization_49/Cast_1/ReadVariableOp,batch_normalization_49/Cast_1/ReadVariableOp2\
,batch_normalization_49/Cast_2/ReadVariableOp,batch_normalization_49/Cast_2/ReadVariableOp2\
,batch_normalization_49/Cast_3/ReadVariableOp,batch_normalization_49/Cast_3/ReadVariableOp2X
*batch_normalization_50/Cast/ReadVariableOp*batch_normalization_50/Cast/ReadVariableOp2\
,batch_normalization_50/Cast_1/ReadVariableOp,batch_normalization_50/Cast_1/ReadVariableOp2\
,batch_normalization_50/Cast_2/ReadVariableOp,batch_normalization_50/Cast_2/ReadVariableOp2\
,batch_normalization_50/Cast_3/ReadVariableOp,batch_normalization_50/Cast_3/ReadVariableOp2X
*batch_normalization_51/Cast/ReadVariableOp*batch_normalization_51/Cast/ReadVariableOp2\
,batch_normalization_51/Cast_1/ReadVariableOp,batch_normalization_51/Cast_1/ReadVariableOp2\
,batch_normalization_51/Cast_2/ReadVariableOp,batch_normalization_51/Cast_2/ReadVariableOp2\
,batch_normalization_51/Cast_3/ReadVariableOp,batch_normalization_51/Cast_3/ReadVariableOp2X
*batch_normalization_52/Cast/ReadVariableOp*batch_normalization_52/Cast/ReadVariableOp2\
,batch_normalization_52/Cast_1/ReadVariableOp,batch_normalization_52/Cast_1/ReadVariableOp2\
,batch_normalization_52/Cast_2/ReadVariableOp,batch_normalization_52/Cast_2/ReadVariableOp2\
,batch_normalization_52/Cast_3/ReadVariableOp,batch_normalization_52/Cast_3/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
�
�
7__inference_batch_normalization_50_layer_call_fn_330372

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
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_3281992
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
R__inference_batch_normalization_49_layer_call_and_return_conditional_losses_328033

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
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_330359

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
)__inference_dense_40_layer_call_fn_330563

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
D__inference_dense_40_layer_call_and_return_conditional_losses_3286962
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
�z
�
"__inference__traced_restore_330813
file_prefixR
Dassignvariableop_feed_forward_sub_net_8_batch_normalization_48_gamma:
S
Eassignvariableop_1_feed_forward_sub_net_8_batch_normalization_48_beta:
T
Fassignvariableop_2_feed_forward_sub_net_8_batch_normalization_49_gamma:S
Eassignvariableop_3_feed_forward_sub_net_8_batch_normalization_49_beta:T
Fassignvariableop_4_feed_forward_sub_net_8_batch_normalization_50_gamma:S
Eassignvariableop_5_feed_forward_sub_net_8_batch_normalization_50_beta:T
Fassignvariableop_6_feed_forward_sub_net_8_batch_normalization_51_gamma:S
Eassignvariableop_7_feed_forward_sub_net_8_batch_normalization_51_beta:T
Fassignvariableop_8_feed_forward_sub_net_8_batch_normalization_52_gamma:S
Eassignvariableop_9_feed_forward_sub_net_8_batch_normalization_52_beta:[
Massignvariableop_10_feed_forward_sub_net_8_batch_normalization_48_moving_mean:
_
Qassignvariableop_11_feed_forward_sub_net_8_batch_normalization_48_moving_variance:
[
Massignvariableop_12_feed_forward_sub_net_8_batch_normalization_49_moving_mean:_
Qassignvariableop_13_feed_forward_sub_net_8_batch_normalization_49_moving_variance:[
Massignvariableop_14_feed_forward_sub_net_8_batch_normalization_50_moving_mean:_
Qassignvariableop_15_feed_forward_sub_net_8_batch_normalization_50_moving_variance:[
Massignvariableop_16_feed_forward_sub_net_8_batch_normalization_51_moving_mean:_
Qassignvariableop_17_feed_forward_sub_net_8_batch_normalization_51_moving_variance:[
Massignvariableop_18_feed_forward_sub_net_8_batch_normalization_52_moving_mean:_
Qassignvariableop_19_feed_forward_sub_net_8_batch_normalization_52_moving_variance:L
:assignvariableop_20_feed_forward_sub_net_8_dense_40_kernel:
L
:assignvariableop_21_feed_forward_sub_net_8_dense_41_kernel:L
:assignvariableop_22_feed_forward_sub_net_8_dense_42_kernel:L
:assignvariableop_23_feed_forward_sub_net_8_dense_43_kernel:L
:assignvariableop_24_feed_forward_sub_net_8_dense_44_kernel:
F
8assignvariableop_25_feed_forward_sub_net_8_dense_44_bias:
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
AssignVariableOpAssignVariableOpDassignvariableop_feed_forward_sub_net_8_batch_normalization_48_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpEassignvariableop_1_feed_forward_sub_net_8_batch_normalization_48_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpFassignvariableop_2_feed_forward_sub_net_8_batch_normalization_49_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpEassignvariableop_3_feed_forward_sub_net_8_batch_normalization_49_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpFassignvariableop_4_feed_forward_sub_net_8_batch_normalization_50_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpEassignvariableop_5_feed_forward_sub_net_8_batch_normalization_50_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpFassignvariableop_6_feed_forward_sub_net_8_batch_normalization_51_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpEassignvariableop_7_feed_forward_sub_net_8_batch_normalization_51_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpFassignvariableop_8_feed_forward_sub_net_8_batch_normalization_52_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpEassignvariableop_9_feed_forward_sub_net_8_batch_normalization_52_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpMassignvariableop_10_feed_forward_sub_net_8_batch_normalization_48_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpQassignvariableop_11_feed_forward_sub_net_8_batch_normalization_48_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpMassignvariableop_12_feed_forward_sub_net_8_batch_normalization_49_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpQassignvariableop_13_feed_forward_sub_net_8_batch_normalization_49_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpMassignvariableop_14_feed_forward_sub_net_8_batch_normalization_50_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpQassignvariableop_15_feed_forward_sub_net_8_batch_normalization_50_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpMassignvariableop_16_feed_forward_sub_net_8_batch_normalization_51_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpQassignvariableop_17_feed_forward_sub_net_8_batch_normalization_51_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpMassignvariableop_18_feed_forward_sub_net_8_batch_normalization_52_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpQassignvariableop_19_feed_forward_sub_net_8_batch_normalization_52_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp:assignvariableop_20_feed_forward_sub_net_8_dense_40_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp:assignvariableop_21_feed_forward_sub_net_8_dense_41_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp:assignvariableop_22_feed_forward_sub_net_8_dense_42_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp:assignvariableop_23_feed_forward_sub_net_8_dense_43_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp:assignvariableop_24_feed_forward_sub_net_8_dense_44_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp8assignvariableop_25_feed_forward_sub_net_8_dense_44_biasIdentity_25:output:0"/device:CPU:0*
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
7__inference_batch_normalization_48_layer_call_fn_330221

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
R__inference_batch_normalization_48_layer_call_and_return_conditional_losses_3279292
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
�
�
)__inference_dense_44_layer_call_fn_330624

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
D__inference_dense_44_layer_call_and_return_conditional_losses_3287832
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
23feed_forward_sub_net_8/batch_normalization_48/gamma
@:>
22feed_forward_sub_net_8/batch_normalization_48/beta
A:?23feed_forward_sub_net_8/batch_normalization_49/gamma
@:>22feed_forward_sub_net_8/batch_normalization_49/beta
A:?23feed_forward_sub_net_8/batch_normalization_50/gamma
@:>22feed_forward_sub_net_8/batch_normalization_50/beta
A:?23feed_forward_sub_net_8/batch_normalization_51/gamma
@:>22feed_forward_sub_net_8/batch_normalization_51/beta
A:?23feed_forward_sub_net_8/batch_normalization_52/gamma
@:>22feed_forward_sub_net_8/batch_normalization_52/beta
I:G
 (29feed_forward_sub_net_8/batch_normalization_48/moving_mean
M:K
 (2=feed_forward_sub_net_8/batch_normalization_48/moving_variance
I:G (29feed_forward_sub_net_8/batch_normalization_49/moving_mean
M:K (2=feed_forward_sub_net_8/batch_normalization_49/moving_variance
I:G (29feed_forward_sub_net_8/batch_normalization_50/moving_mean
M:K (2=feed_forward_sub_net_8/batch_normalization_50/moving_variance
I:G (29feed_forward_sub_net_8/batch_normalization_51/moving_mean
M:K (2=feed_forward_sub_net_8/batch_normalization_51/moving_variance
I:G (29feed_forward_sub_net_8/batch_normalization_52/moving_mean
M:K (2=feed_forward_sub_net_8/batch_normalization_52/moving_variance
8:6
2&feed_forward_sub_net_8/dense_40/kernel
8:62&feed_forward_sub_net_8/dense_41/kernel
8:62&feed_forward_sub_net_8/dense_42/kernel
8:62&feed_forward_sub_net_8/dense_43/kernel
8:6
2&feed_forward_sub_net_8/dense_44/kernel
2:0
2$feed_forward_sub_net_8/dense_44/bias
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
!__inference__wrapped_model_327843input_1"�
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
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329433
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329619
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329725
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329911�
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
7__inference_feed_forward_sub_net_8_layer_call_fn_329968
7__inference_feed_forward_sub_net_8_layer_call_fn_330025
7__inference_feed_forward_sub_net_8_layer_call_fn_330082
7__inference_feed_forward_sub_net_8_layer_call_fn_330139�
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
$__inference_signature_wrapper_329327input_1"�
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
R__inference_batch_normalization_48_layer_call_and_return_conditional_losses_330159
R__inference_batch_normalization_48_layer_call_and_return_conditional_losses_330195�
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
7__inference_batch_normalization_48_layer_call_fn_330208
7__inference_batch_normalization_48_layer_call_fn_330221�
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
R__inference_batch_normalization_49_layer_call_and_return_conditional_losses_330241
R__inference_batch_normalization_49_layer_call_and_return_conditional_losses_330277�
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
7__inference_batch_normalization_49_layer_call_fn_330290
7__inference_batch_normalization_49_layer_call_fn_330303�
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
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_330323
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_330359�
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
7__inference_batch_normalization_50_layer_call_fn_330372
7__inference_batch_normalization_50_layer_call_fn_330385�
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
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_330405
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_330441�
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
7__inference_batch_normalization_51_layer_call_fn_330454
7__inference_batch_normalization_51_layer_call_fn_330467�
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
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_330487
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_330523�
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
7__inference_batch_normalization_52_layer_call_fn_330536
7__inference_batch_normalization_52_layer_call_fn_330549�
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
D__inference_dense_40_layer_call_and_return_conditional_losses_330556�
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
)__inference_dense_40_layer_call_fn_330563�
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
D__inference_dense_41_layer_call_and_return_conditional_losses_330570�
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
)__inference_dense_41_layer_call_fn_330577�
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
D__inference_dense_42_layer_call_and_return_conditional_losses_330584�
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
)__inference_dense_42_layer_call_fn_330591�
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
D__inference_dense_43_layer_call_and_return_conditional_losses_330598�
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
)__inference_dense_43_layer_call_fn_330605�
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
D__inference_dense_44_layer_call_and_return_conditional_losses_330615�
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
)__inference_dense_44_layer_call_fn_330624�
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
!__inference__wrapped_model_327843�' (!")#$*%&+,0�-
&�#
!�
input_1���������

� "3�0
.
output_1"�
output_1���������
�
R__inference_batch_normalization_48_layer_call_and_return_conditional_losses_330159b3�0
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
R__inference_batch_normalization_48_layer_call_and_return_conditional_losses_330195b3�0
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
7__inference_batch_normalization_48_layer_call_fn_330208U3�0
)�&
 �
inputs���������

p 
� "����������
�
7__inference_batch_normalization_48_layer_call_fn_330221U3�0
)�&
 �
inputs���������

p
� "����������
�
R__inference_batch_normalization_49_layer_call_and_return_conditional_losses_330241b 3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_49_layer_call_and_return_conditional_losses_330277b 3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_49_layer_call_fn_330290U 3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_49_layer_call_fn_330303U 3�0
)�&
 �
inputs���������
p
� "�����������
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_330323b!"3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_50_layer_call_and_return_conditional_losses_330359b!"3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_50_layer_call_fn_330372U!"3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_50_layer_call_fn_330385U!"3�0
)�&
 �
inputs���������
p
� "�����������
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_330405b#$3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_51_layer_call_and_return_conditional_losses_330441b#$3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_51_layer_call_fn_330454U#$3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_51_layer_call_fn_330467U#$3�0
)�&
 �
inputs���������
p
� "�����������
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_330487b%&3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_52_layer_call_and_return_conditional_losses_330523b%&3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_52_layer_call_fn_330536U%&3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_52_layer_call_fn_330549U%&3�0
)�&
 �
inputs���������
p
� "�����������
D__inference_dense_40_layer_call_and_return_conditional_losses_330556['/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� {
)__inference_dense_40_layer_call_fn_330563N'/�,
%�"
 �
inputs���������

� "�����������
D__inference_dense_41_layer_call_and_return_conditional_losses_330570[(/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
)__inference_dense_41_layer_call_fn_330577N(/�,
%�"
 �
inputs���������
� "�����������
D__inference_dense_42_layer_call_and_return_conditional_losses_330584[)/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
)__inference_dense_42_layer_call_fn_330591N)/�,
%�"
 �
inputs���������
� "�����������
D__inference_dense_43_layer_call_and_return_conditional_losses_330598[*/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
)__inference_dense_43_layer_call_fn_330605N*/�,
%�"
 �
inputs���������
� "�����������
D__inference_dense_44_layer_call_and_return_conditional_losses_330615\+,/�,
%�"
 �
inputs���������
� "%�"
�
0���������

� |
)__inference_dense_44_layer_call_fn_330624O+,/�,
%�"
 �
inputs���������
� "����������
�
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329433s' (!")#$*%&+,.�+
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
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329619s' (!")#$*%&+,.�+
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
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329725y' (!")#$*%&+,4�1
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
R__inference_feed_forward_sub_net_8_layer_call_and_return_conditional_losses_329911y' (!")#$*%&+,4�1
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
7__inference_feed_forward_sub_net_8_layer_call_fn_329968l' (!")#$*%&+,4�1
*�'
!�
input_1���������

p 
� "����������
�
7__inference_feed_forward_sub_net_8_layer_call_fn_330025f' (!")#$*%&+,.�+
$�!
�
x���������

p 
� "����������
�
7__inference_feed_forward_sub_net_8_layer_call_fn_330082f' (!")#$*%&+,.�+
$�!
�
x���������

p
� "����������
�
7__inference_feed_forward_sub_net_8_layer_call_fn_330139l' (!")#$*%&+,4�1
*�'
!�
input_1���������

p
� "����������
�
$__inference_signature_wrapper_329327�' (!")#$*%&+,;�8
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