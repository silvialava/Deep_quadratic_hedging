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
3feed_forward_sub_net_2/batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*D
shared_name53feed_forward_sub_net_2/batch_normalization_12/gamma
�
Gfeed_forward_sub_net_2/batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_2/batch_normalization_12/gamma*
_output_shapes
:
*
dtype0
�
2feed_forward_sub_net_2/batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*C
shared_name42feed_forward_sub_net_2/batch_normalization_12/beta
�
Ffeed_forward_sub_net_2/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_2/batch_normalization_12/beta*
_output_shapes
:
*
dtype0
�
3feed_forward_sub_net_2/batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_2/batch_normalization_13/gamma
�
Gfeed_forward_sub_net_2/batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_2/batch_normalization_13/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_2/batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_2/batch_normalization_13/beta
�
Ffeed_forward_sub_net_2/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_2/batch_normalization_13/beta*
_output_shapes
:*
dtype0
�
3feed_forward_sub_net_2/batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_2/batch_normalization_14/gamma
�
Gfeed_forward_sub_net_2/batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_2/batch_normalization_14/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_2/batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_2/batch_normalization_14/beta
�
Ffeed_forward_sub_net_2/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_2/batch_normalization_14/beta*
_output_shapes
:*
dtype0
�
3feed_forward_sub_net_2/batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_2/batch_normalization_15/gamma
�
Gfeed_forward_sub_net_2/batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_2/batch_normalization_15/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_2/batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_2/batch_normalization_15/beta
�
Ffeed_forward_sub_net_2/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_2/batch_normalization_15/beta*
_output_shapes
:*
dtype0
�
3feed_forward_sub_net_2/batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_2/batch_normalization_16/gamma
�
Gfeed_forward_sub_net_2/batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_2/batch_normalization_16/gamma*
_output_shapes
:*
dtype0
�
2feed_forward_sub_net_2/batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_2/batch_normalization_16/beta
�
Ffeed_forward_sub_net_2/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_2/batch_normalization_16/beta*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_2/batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*J
shared_name;9feed_forward_sub_net_2/batch_normalization_12/moving_mean
�
Mfeed_forward_sub_net_2/batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_2/batch_normalization_12/moving_mean*
_output_shapes
:
*
dtype0
�
=feed_forward_sub_net_2/batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*N
shared_name?=feed_forward_sub_net_2/batch_normalization_12/moving_variance
�
Qfeed_forward_sub_net_2/batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_2/batch_normalization_12/moving_variance*
_output_shapes
:
*
dtype0
�
9feed_forward_sub_net_2/batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_2/batch_normalization_13/moving_mean
�
Mfeed_forward_sub_net_2/batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_2/batch_normalization_13/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_2/batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_2/batch_normalization_13/moving_variance
�
Qfeed_forward_sub_net_2/batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_2/batch_normalization_13/moving_variance*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_2/batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_2/batch_normalization_14/moving_mean
�
Mfeed_forward_sub_net_2/batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_2/batch_normalization_14/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_2/batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_2/batch_normalization_14/moving_variance
�
Qfeed_forward_sub_net_2/batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_2/batch_normalization_14/moving_variance*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_2/batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_2/batch_normalization_15/moving_mean
�
Mfeed_forward_sub_net_2/batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_2/batch_normalization_15/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_2/batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_2/batch_normalization_15/moving_variance
�
Qfeed_forward_sub_net_2/batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_2/batch_normalization_15/moving_variance*
_output_shapes
:*
dtype0
�
9feed_forward_sub_net_2/batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_2/batch_normalization_16/moving_mean
�
Mfeed_forward_sub_net_2/batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_2/batch_normalization_16/moving_mean*
_output_shapes
:*
dtype0
�
=feed_forward_sub_net_2/batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_2/batch_normalization_16/moving_variance
�
Qfeed_forward_sub_net_2/batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_2/batch_normalization_16/moving_variance*
_output_shapes
:*
dtype0
�
&feed_forward_sub_net_2/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&feed_forward_sub_net_2/dense_10/kernel
�
:feed_forward_sub_net_2/dense_10/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_2/dense_10/kernel*
_output_shapes

:
*
dtype0
�
&feed_forward_sub_net_2/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_2/dense_11/kernel
�
:feed_forward_sub_net_2/dense_11/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_2/dense_11/kernel*
_output_shapes

:*
dtype0
�
&feed_forward_sub_net_2/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_2/dense_12/kernel
�
:feed_forward_sub_net_2/dense_12/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_2/dense_12/kernel*
_output_shapes

:*
dtype0
�
&feed_forward_sub_net_2/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_2/dense_13/kernel
�
:feed_forward_sub_net_2/dense_13/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_2/dense_13/kernel*
_output_shapes

:*
dtype0
�
&feed_forward_sub_net_2/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&feed_forward_sub_net_2/dense_14/kernel
�
:feed_forward_sub_net_2/dense_14/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_2/dense_14/kernel*
_output_shapes

:
*
dtype0
�
$feed_forward_sub_net_2/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$feed_forward_sub_net_2/dense_14/bias
�
8feed_forward_sub_net_2/dense_14/bias/Read/ReadVariableOpReadVariableOp$feed_forward_sub_net_2/dense_14/bias*
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
regularization_losses
	variables
trainable_variables
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
 
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
�
-layer_metrics
regularization_losses
.metrics
	variables
/layer_regularization_losses

0layers
1non_trainable_variables
trainable_variables
 
�
2axis
	gamma
beta
moving_mean
moving_variance
3regularization_losses
4	variables
5trainable_variables
6	keras_api
�
7axis
	gamma
beta
moving_mean
 moving_variance
8regularization_losses
9	variables
:trainable_variables
;	keras_api
�
<axis
	gamma
beta
!moving_mean
"moving_variance
=regularization_losses
>	variables
?trainable_variables
@	keras_api
�
Aaxis
	gamma
beta
#moving_mean
$moving_variance
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
�
Faxis
	gamma
beta
%moving_mean
&moving_variance
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api

K	keras_api
^

'kernel
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
^

(kernel
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
^

)kernel
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
^

*kernel
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
h

+kernel
,bias
\regularization_losses
]	variables
^trainable_variables
_	keras_api
om
VARIABLE_VALUE3feed_forward_sub_net_2/batch_normalization_12/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_2/batch_normalization_12/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_2/batch_normalization_13/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_2/batch_normalization_13/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_2/batch_normalization_14/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_2/batch_normalization_14/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_2/batch_normalization_15/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_2/batch_normalization_15/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_2/batch_normalization_16/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_2/batch_normalization_16/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_2/batch_normalization_12/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_2/batch_normalization_12/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_2/batch_normalization_13/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_2/batch_normalization_13/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_2/batch_normalization_14/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_2/batch_normalization_14/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_2/batch_normalization_15/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_2/batch_normalization_15/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_2/batch_normalization_16/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_2/batch_normalization_16/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_2/dense_10/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_2/dense_11/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_2/dense_12/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_2/dense_13/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_2/dense_14/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$feed_forward_sub_net_2/dense_14/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
 
 
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

0
1
2
3

0
1
�
`layer_metrics
3regularization_losses
ametrics
4	variables
blayer_regularization_losses

clayers
dnon_trainable_variables
5trainable_variables
 
 

0
1
2
 3

0
1
�
elayer_metrics
8regularization_losses
fmetrics
9	variables
glayer_regularization_losses

hlayers
inon_trainable_variables
:trainable_variables
 
 

0
1
!2
"3

0
1
�
jlayer_metrics
=regularization_losses
kmetrics
>	variables
llayer_regularization_losses

mlayers
nnon_trainable_variables
?trainable_variables
 
 

0
1
#2
$3

0
1
�
olayer_metrics
Bregularization_losses
pmetrics
C	variables
qlayer_regularization_losses

rlayers
snon_trainable_variables
Dtrainable_variables
 
 

0
1
%2
&3

0
1
�
tlayer_metrics
Gregularization_losses
umetrics
H	variables
vlayer_regularization_losses

wlayers
xnon_trainable_variables
Itrainable_variables
 
 

'0

'0
�
ylayer_metrics
Lregularization_losses
zmetrics
M	variables
{layer_regularization_losses

|layers
}non_trainable_variables
Ntrainable_variables
 

(0

(0
�
~layer_metrics
Pregularization_losses
metrics
Q	variables
 �layer_regularization_losses
�layers
�non_trainable_variables
Rtrainable_variables
 

)0

)0
�
�layer_metrics
Tregularization_losses
�metrics
U	variables
 �layer_regularization_losses
�layers
�non_trainable_variables
Vtrainable_variables
 

*0

*0
�
�layer_metrics
Xregularization_losses
�metrics
Y	variables
 �layer_regularization_losses
�layers
�non_trainable_variables
Ztrainable_variables
 

+0
,1

+0
,1
�
�layer_metrics
\regularization_losses
�metrics
]	variables
 �layer_regularization_losses
�layers
�non_trainable_variables
^trainable_variables
 
 
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
z
serving_default_input_1Placeholder*'
_output_shapes
:���������
*
dtype0*
shape:���������

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_19feed_forward_sub_net_2/batch_normalization_12/moving_mean=feed_forward_sub_net_2/batch_normalization_12/moving_variance2feed_forward_sub_net_2/batch_normalization_12/beta3feed_forward_sub_net_2/batch_normalization_12/gamma&feed_forward_sub_net_2/dense_10/kernel9feed_forward_sub_net_2/batch_normalization_13/moving_mean=feed_forward_sub_net_2/batch_normalization_13/moving_variance2feed_forward_sub_net_2/batch_normalization_13/beta3feed_forward_sub_net_2/batch_normalization_13/gamma&feed_forward_sub_net_2/dense_11/kernel9feed_forward_sub_net_2/batch_normalization_14/moving_mean=feed_forward_sub_net_2/batch_normalization_14/moving_variance2feed_forward_sub_net_2/batch_normalization_14/beta3feed_forward_sub_net_2/batch_normalization_14/gamma&feed_forward_sub_net_2/dense_12/kernel9feed_forward_sub_net_2/batch_normalization_15/moving_mean=feed_forward_sub_net_2/batch_normalization_15/moving_variance2feed_forward_sub_net_2/batch_normalization_15/beta3feed_forward_sub_net_2/batch_normalization_15/gamma&feed_forward_sub_net_2/dense_13/kernel9feed_forward_sub_net_2/batch_normalization_16/moving_mean=feed_forward_sub_net_2/batch_normalization_16/moving_variance2feed_forward_sub_net_2/batch_normalization_16/beta3feed_forward_sub_net_2/batch_normalization_16/gamma&feed_forward_sub_net_2/dense_14/kernel$feed_forward_sub_net_2/dense_14/bias*&
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
$__inference_signature_wrapper_314645
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameGfeed_forward_sub_net_2/batch_normalization_12/gamma/Read/ReadVariableOpFfeed_forward_sub_net_2/batch_normalization_12/beta/Read/ReadVariableOpGfeed_forward_sub_net_2/batch_normalization_13/gamma/Read/ReadVariableOpFfeed_forward_sub_net_2/batch_normalization_13/beta/Read/ReadVariableOpGfeed_forward_sub_net_2/batch_normalization_14/gamma/Read/ReadVariableOpFfeed_forward_sub_net_2/batch_normalization_14/beta/Read/ReadVariableOpGfeed_forward_sub_net_2/batch_normalization_15/gamma/Read/ReadVariableOpFfeed_forward_sub_net_2/batch_normalization_15/beta/Read/ReadVariableOpGfeed_forward_sub_net_2/batch_normalization_16/gamma/Read/ReadVariableOpFfeed_forward_sub_net_2/batch_normalization_16/beta/Read/ReadVariableOpMfeed_forward_sub_net_2/batch_normalization_12/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_2/batch_normalization_12/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_2/batch_normalization_13/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_2/batch_normalization_13/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_2/batch_normalization_14/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_2/batch_normalization_14/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_2/batch_normalization_15/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_2/batch_normalization_15/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_2/batch_normalization_16/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_2/batch_normalization_16/moving_variance/Read/ReadVariableOp:feed_forward_sub_net_2/dense_10/kernel/Read/ReadVariableOp:feed_forward_sub_net_2/dense_11/kernel/Read/ReadVariableOp:feed_forward_sub_net_2/dense_12/kernel/Read/ReadVariableOp:feed_forward_sub_net_2/dense_13/kernel/Read/ReadVariableOp:feed_forward_sub_net_2/dense_14/kernel/Read/ReadVariableOp8feed_forward_sub_net_2/dense_14/bias/Read/ReadVariableOpConst*'
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
__inference__traced_save_316043
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename3feed_forward_sub_net_2/batch_normalization_12/gamma2feed_forward_sub_net_2/batch_normalization_12/beta3feed_forward_sub_net_2/batch_normalization_13/gamma2feed_forward_sub_net_2/batch_normalization_13/beta3feed_forward_sub_net_2/batch_normalization_14/gamma2feed_forward_sub_net_2/batch_normalization_14/beta3feed_forward_sub_net_2/batch_normalization_15/gamma2feed_forward_sub_net_2/batch_normalization_15/beta3feed_forward_sub_net_2/batch_normalization_16/gamma2feed_forward_sub_net_2/batch_normalization_16/beta9feed_forward_sub_net_2/batch_normalization_12/moving_mean=feed_forward_sub_net_2/batch_normalization_12/moving_variance9feed_forward_sub_net_2/batch_normalization_13/moving_mean=feed_forward_sub_net_2/batch_normalization_13/moving_variance9feed_forward_sub_net_2/batch_normalization_14/moving_mean=feed_forward_sub_net_2/batch_normalization_14/moving_variance9feed_forward_sub_net_2/batch_normalization_15/moving_mean=feed_forward_sub_net_2/batch_normalization_15/moving_variance9feed_forward_sub_net_2/batch_normalization_16/moving_mean=feed_forward_sub_net_2/batch_normalization_16/moving_variance&feed_forward_sub_net_2/dense_10/kernel&feed_forward_sub_net_2/dense_11/kernel&feed_forward_sub_net_2/dense_12/kernel&feed_forward_sub_net_2/dense_13/kernel&feed_forward_sub_net_2/dense_14/kernel$feed_forward_sub_net_2/dense_14/bias*&
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
"__inference__traced_restore_316131��
�
}
)__inference_dense_11_layer_call_fn_315888

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
D__inference_dense_11_layer_call_and_return_conditional_losses_3140352
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
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_313413

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
7__inference_batch_normalization_12_layer_call_fn_315470

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
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3131852
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
�
�
7__inference_batch_normalization_14_layer_call_fn_315647

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
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_3135792
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
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_313849

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
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_313517

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
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_313185

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
�
}
)__inference_dense_13_layer_call_fn_315916

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
D__inference_dense_13_layer_call_and_return_conditional_losses_3140772
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
��
� 
!__inference__wrapped_model_313161
input_1X
Jfeed_forward_sub_net_2_batch_normalization_12_cast_readvariableop_resource:
Z
Lfeed_forward_sub_net_2_batch_normalization_12_cast_1_readvariableop_resource:
Z
Lfeed_forward_sub_net_2_batch_normalization_12_cast_2_readvariableop_resource:
Z
Lfeed_forward_sub_net_2_batch_normalization_12_cast_3_readvariableop_resource:
P
>feed_forward_sub_net_2_dense_10_matmul_readvariableop_resource:
X
Jfeed_forward_sub_net_2_batch_normalization_13_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_2_batch_normalization_13_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_2_batch_normalization_13_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_2_batch_normalization_13_cast_3_readvariableop_resource:P
>feed_forward_sub_net_2_dense_11_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_2_batch_normalization_14_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_2_batch_normalization_14_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_2_batch_normalization_14_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_2_batch_normalization_14_cast_3_readvariableop_resource:P
>feed_forward_sub_net_2_dense_12_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_2_batch_normalization_15_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_2_batch_normalization_15_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_2_batch_normalization_15_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_2_batch_normalization_15_cast_3_readvariableop_resource:P
>feed_forward_sub_net_2_dense_13_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_2_batch_normalization_16_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_2_batch_normalization_16_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_2_batch_normalization_16_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_2_batch_normalization_16_cast_3_readvariableop_resource:P
>feed_forward_sub_net_2_dense_14_matmul_readvariableop_resource:
M
?feed_forward_sub_net_2_dense_14_biasadd_readvariableop_resource:

identity��Afeed_forward_sub_net_2/batch_normalization_12/Cast/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_12/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_12/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_12/Cast_3/ReadVariableOp�Afeed_forward_sub_net_2/batch_normalization_13/Cast/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_13/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_13/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_13/Cast_3/ReadVariableOp�Afeed_forward_sub_net_2/batch_normalization_14/Cast/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_14/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_14/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_14/Cast_3/ReadVariableOp�Afeed_forward_sub_net_2/batch_normalization_15/Cast/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_15/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_15/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_15/Cast_3/ReadVariableOp�Afeed_forward_sub_net_2/batch_normalization_16/Cast/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_16/Cast_1/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_16/Cast_2/ReadVariableOp�Cfeed_forward_sub_net_2/batch_normalization_16/Cast_3/ReadVariableOp�5feed_forward_sub_net_2/dense_10/MatMul/ReadVariableOp�5feed_forward_sub_net_2/dense_11/MatMul/ReadVariableOp�5feed_forward_sub_net_2/dense_12/MatMul/ReadVariableOp�5feed_forward_sub_net_2/dense_13/MatMul/ReadVariableOp�6feed_forward_sub_net_2/dense_14/BiasAdd/ReadVariableOp�5feed_forward_sub_net_2/dense_14/MatMul/ReadVariableOp�
Afeed_forward_sub_net_2/batch_normalization_12/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_2_batch_normalization_12_cast_readvariableop_resource*
_output_shapes
:
*
dtype02C
Afeed_forward_sub_net_2/batch_normalization_12/Cast/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_12/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_12_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_12/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_12/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_12_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_12/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_12/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_12_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_12/Cast_3/ReadVariableOp�
=feed_forward_sub_net_2/batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_2/batch_normalization_12/batchnorm/add/y�
;feed_forward_sub_net_2/batch_normalization_12/batchnorm/addAddV2Kfeed_forward_sub_net_2/batch_normalization_12/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_2/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_2/batch_normalization_12/batchnorm/add�
=feed_forward_sub_net_2/batch_normalization_12/batchnorm/RsqrtRsqrt?feed_forward_sub_net_2/batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_2/batch_normalization_12/batchnorm/Rsqrt�
;feed_forward_sub_net_2/batch_normalization_12/batchnorm/mulMulAfeed_forward_sub_net_2/batch_normalization_12/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_2/batch_normalization_12/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_2/batch_normalization_12/batchnorm/mul�
=feed_forward_sub_net_2/batch_normalization_12/batchnorm/mul_1Mulinput_1?feed_forward_sub_net_2/batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2?
=feed_forward_sub_net_2/batch_normalization_12/batchnorm/mul_1�
=feed_forward_sub_net_2/batch_normalization_12/batchnorm/mul_2MulIfeed_forward_sub_net_2/batch_normalization_12/Cast/ReadVariableOp:value:0?feed_forward_sub_net_2/batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_2/batch_normalization_12/batchnorm/mul_2�
;feed_forward_sub_net_2/batch_normalization_12/batchnorm/subSubKfeed_forward_sub_net_2/batch_normalization_12/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_2/batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_2/batch_normalization_12/batchnorm/sub�
=feed_forward_sub_net_2/batch_normalization_12/batchnorm/add_1AddV2Afeed_forward_sub_net_2/batch_normalization_12/batchnorm/mul_1:z:0?feed_forward_sub_net_2/batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2?
=feed_forward_sub_net_2/batch_normalization_12/batchnorm/add_1�
5feed_forward_sub_net_2/dense_10/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5feed_forward_sub_net_2/dense_10/MatMul/ReadVariableOp�
&feed_forward_sub_net_2/dense_10/MatMulMatMulAfeed_forward_sub_net_2/batch_normalization_12/batchnorm/add_1:z:0=feed_forward_sub_net_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&feed_forward_sub_net_2/dense_10/MatMul�
Afeed_forward_sub_net_2/batch_normalization_13/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_2_batch_normalization_13_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_2/batch_normalization_13/Cast/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_13/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_13_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_13/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_13/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_13_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_13/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_13/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_13_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_13/Cast_3/ReadVariableOp�
=feed_forward_sub_net_2/batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_2/batch_normalization_13/batchnorm/add/y�
;feed_forward_sub_net_2/batch_normalization_13/batchnorm/addAddV2Kfeed_forward_sub_net_2/batch_normalization_13/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_2/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_2/batch_normalization_13/batchnorm/add�
=feed_forward_sub_net_2/batch_normalization_13/batchnorm/RsqrtRsqrt?feed_forward_sub_net_2/batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_2/batch_normalization_13/batchnorm/Rsqrt�
;feed_forward_sub_net_2/batch_normalization_13/batchnorm/mulMulAfeed_forward_sub_net_2/batch_normalization_13/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_2/batch_normalization_13/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_2/batch_normalization_13/batchnorm/mul�
=feed_forward_sub_net_2/batch_normalization_13/batchnorm/mul_1Mul0feed_forward_sub_net_2/dense_10/MatMul:product:0?feed_forward_sub_net_2/batch_normalization_13/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_2/batch_normalization_13/batchnorm/mul_1�
=feed_forward_sub_net_2/batch_normalization_13/batchnorm/mul_2MulIfeed_forward_sub_net_2/batch_normalization_13/Cast/ReadVariableOp:value:0?feed_forward_sub_net_2/batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_2/batch_normalization_13/batchnorm/mul_2�
;feed_forward_sub_net_2/batch_normalization_13/batchnorm/subSubKfeed_forward_sub_net_2/batch_normalization_13/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_2/batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_2/batch_normalization_13/batchnorm/sub�
=feed_forward_sub_net_2/batch_normalization_13/batchnorm/add_1AddV2Afeed_forward_sub_net_2/batch_normalization_13/batchnorm/mul_1:z:0?feed_forward_sub_net_2/batch_normalization_13/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_2/batch_normalization_13/batchnorm/add_1�
feed_forward_sub_net_2/ReluReluAfeed_forward_sub_net_2/batch_normalization_13/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_2/Relu�
5feed_forward_sub_net_2/dense_11/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_2_dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_2/dense_11/MatMul/ReadVariableOp�
&feed_forward_sub_net_2/dense_11/MatMulMatMul)feed_forward_sub_net_2/Relu:activations:0=feed_forward_sub_net_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&feed_forward_sub_net_2/dense_11/MatMul�
Afeed_forward_sub_net_2/batch_normalization_14/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_2_batch_normalization_14_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_2/batch_normalization_14/Cast/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_14/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_14_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_14/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_14/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_14_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_14/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_14/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_14_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_14/Cast_3/ReadVariableOp�
=feed_forward_sub_net_2/batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_2/batch_normalization_14/batchnorm/add/y�
;feed_forward_sub_net_2/batch_normalization_14/batchnorm/addAddV2Kfeed_forward_sub_net_2/batch_normalization_14/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_2/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_2/batch_normalization_14/batchnorm/add�
=feed_forward_sub_net_2/batch_normalization_14/batchnorm/RsqrtRsqrt?feed_forward_sub_net_2/batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_2/batch_normalization_14/batchnorm/Rsqrt�
;feed_forward_sub_net_2/batch_normalization_14/batchnorm/mulMulAfeed_forward_sub_net_2/batch_normalization_14/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_2/batch_normalization_14/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_2/batch_normalization_14/batchnorm/mul�
=feed_forward_sub_net_2/batch_normalization_14/batchnorm/mul_1Mul0feed_forward_sub_net_2/dense_11/MatMul:product:0?feed_forward_sub_net_2/batch_normalization_14/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_2/batch_normalization_14/batchnorm/mul_1�
=feed_forward_sub_net_2/batch_normalization_14/batchnorm/mul_2MulIfeed_forward_sub_net_2/batch_normalization_14/Cast/ReadVariableOp:value:0?feed_forward_sub_net_2/batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_2/batch_normalization_14/batchnorm/mul_2�
;feed_forward_sub_net_2/batch_normalization_14/batchnorm/subSubKfeed_forward_sub_net_2/batch_normalization_14/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_2/batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_2/batch_normalization_14/batchnorm/sub�
=feed_forward_sub_net_2/batch_normalization_14/batchnorm/add_1AddV2Afeed_forward_sub_net_2/batch_normalization_14/batchnorm/mul_1:z:0?feed_forward_sub_net_2/batch_normalization_14/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_2/batch_normalization_14/batchnorm/add_1�
feed_forward_sub_net_2/Relu_1ReluAfeed_forward_sub_net_2/batch_normalization_14/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_2/Relu_1�
5feed_forward_sub_net_2/dense_12/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_2/dense_12/MatMul/ReadVariableOp�
&feed_forward_sub_net_2/dense_12/MatMulMatMul+feed_forward_sub_net_2/Relu_1:activations:0=feed_forward_sub_net_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&feed_forward_sub_net_2/dense_12/MatMul�
Afeed_forward_sub_net_2/batch_normalization_15/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_2_batch_normalization_15_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_2/batch_normalization_15/Cast/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_15/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_15_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_15/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_15/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_15_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_15/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_15/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_15_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_15/Cast_3/ReadVariableOp�
=feed_forward_sub_net_2/batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_2/batch_normalization_15/batchnorm/add/y�
;feed_forward_sub_net_2/batch_normalization_15/batchnorm/addAddV2Kfeed_forward_sub_net_2/batch_normalization_15/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_2/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_2/batch_normalization_15/batchnorm/add�
=feed_forward_sub_net_2/batch_normalization_15/batchnorm/RsqrtRsqrt?feed_forward_sub_net_2/batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_2/batch_normalization_15/batchnorm/Rsqrt�
;feed_forward_sub_net_2/batch_normalization_15/batchnorm/mulMulAfeed_forward_sub_net_2/batch_normalization_15/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_2/batch_normalization_15/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_2/batch_normalization_15/batchnorm/mul�
=feed_forward_sub_net_2/batch_normalization_15/batchnorm/mul_1Mul0feed_forward_sub_net_2/dense_12/MatMul:product:0?feed_forward_sub_net_2/batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_2/batch_normalization_15/batchnorm/mul_1�
=feed_forward_sub_net_2/batch_normalization_15/batchnorm/mul_2MulIfeed_forward_sub_net_2/batch_normalization_15/Cast/ReadVariableOp:value:0?feed_forward_sub_net_2/batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_2/batch_normalization_15/batchnorm/mul_2�
;feed_forward_sub_net_2/batch_normalization_15/batchnorm/subSubKfeed_forward_sub_net_2/batch_normalization_15/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_2/batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_2/batch_normalization_15/batchnorm/sub�
=feed_forward_sub_net_2/batch_normalization_15/batchnorm/add_1AddV2Afeed_forward_sub_net_2/batch_normalization_15/batchnorm/mul_1:z:0?feed_forward_sub_net_2/batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_2/batch_normalization_15/batchnorm/add_1�
feed_forward_sub_net_2/Relu_2ReluAfeed_forward_sub_net_2/batch_normalization_15/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_2/Relu_2�
5feed_forward_sub_net_2/dense_13/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_2/dense_13/MatMul/ReadVariableOp�
&feed_forward_sub_net_2/dense_13/MatMulMatMul+feed_forward_sub_net_2/Relu_2:activations:0=feed_forward_sub_net_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&feed_forward_sub_net_2/dense_13/MatMul�
Afeed_forward_sub_net_2/batch_normalization_16/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_2_batch_normalization_16_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_2/batch_normalization_16/Cast/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_16/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_16/Cast_1/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_16/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_16_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_16/Cast_2/ReadVariableOp�
Cfeed_forward_sub_net_2/batch_normalization_16/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_2_batch_normalization_16_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_2/batch_normalization_16/Cast_3/ReadVariableOp�
=feed_forward_sub_net_2/batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2?
=feed_forward_sub_net_2/batch_normalization_16/batchnorm/add/y�
;feed_forward_sub_net_2/batch_normalization_16/batchnorm/addAddV2Kfeed_forward_sub_net_2/batch_normalization_16/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_2/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_2/batch_normalization_16/batchnorm/add�
=feed_forward_sub_net_2/batch_normalization_16/batchnorm/RsqrtRsqrt?feed_forward_sub_net_2/batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_2/batch_normalization_16/batchnorm/Rsqrt�
;feed_forward_sub_net_2/batch_normalization_16/batchnorm/mulMulAfeed_forward_sub_net_2/batch_normalization_16/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_2/batch_normalization_16/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_2/batch_normalization_16/batchnorm/mul�
=feed_forward_sub_net_2/batch_normalization_16/batchnorm/mul_1Mul0feed_forward_sub_net_2/dense_13/MatMul:product:0?feed_forward_sub_net_2/batch_normalization_16/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_2/batch_normalization_16/batchnorm/mul_1�
=feed_forward_sub_net_2/batch_normalization_16/batchnorm/mul_2MulIfeed_forward_sub_net_2/batch_normalization_16/Cast/ReadVariableOp:value:0?feed_forward_sub_net_2/batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_2/batch_normalization_16/batchnorm/mul_2�
;feed_forward_sub_net_2/batch_normalization_16/batchnorm/subSubKfeed_forward_sub_net_2/batch_normalization_16/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_2/batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_2/batch_normalization_16/batchnorm/sub�
=feed_forward_sub_net_2/batch_normalization_16/batchnorm/add_1AddV2Afeed_forward_sub_net_2/batch_normalization_16/batchnorm/mul_1:z:0?feed_forward_sub_net_2/batch_normalization_16/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2?
=feed_forward_sub_net_2/batch_normalization_16/batchnorm/add_1�
feed_forward_sub_net_2/Relu_3ReluAfeed_forward_sub_net_2/batch_normalization_16/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
feed_forward_sub_net_2/Relu_3�
5feed_forward_sub_net_2/dense_14/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_2_dense_14_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5feed_forward_sub_net_2/dense_14/MatMul/ReadVariableOp�
&feed_forward_sub_net_2/dense_14/MatMulMatMul+feed_forward_sub_net_2/Relu_3:activations:0=feed_forward_sub_net_2/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2(
&feed_forward_sub_net_2/dense_14/MatMul�
6feed_forward_sub_net_2/dense_14/BiasAdd/ReadVariableOpReadVariableOp?feed_forward_sub_net_2_dense_14_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype028
6feed_forward_sub_net_2/dense_14/BiasAdd/ReadVariableOp�
'feed_forward_sub_net_2/dense_14/BiasAddBiasAdd0feed_forward_sub_net_2/dense_14/MatMul:product:0>feed_forward_sub_net_2/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2)
'feed_forward_sub_net_2/dense_14/BiasAdd�
IdentityIdentity0feed_forward_sub_net_2/dense_14/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOpB^feed_forward_sub_net_2/batch_normalization_12/Cast/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_12/Cast_1/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_12/Cast_2/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_12/Cast_3/ReadVariableOpB^feed_forward_sub_net_2/batch_normalization_13/Cast/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_13/Cast_1/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_13/Cast_2/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_13/Cast_3/ReadVariableOpB^feed_forward_sub_net_2/batch_normalization_14/Cast/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_14/Cast_1/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_14/Cast_2/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_14/Cast_3/ReadVariableOpB^feed_forward_sub_net_2/batch_normalization_15/Cast/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_15/Cast_1/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_15/Cast_2/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_15/Cast_3/ReadVariableOpB^feed_forward_sub_net_2/batch_normalization_16/Cast/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_16/Cast_1/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_16/Cast_2/ReadVariableOpD^feed_forward_sub_net_2/batch_normalization_16/Cast_3/ReadVariableOp6^feed_forward_sub_net_2/dense_10/MatMul/ReadVariableOp6^feed_forward_sub_net_2/dense_11/MatMul/ReadVariableOp6^feed_forward_sub_net_2/dense_12/MatMul/ReadVariableOp6^feed_forward_sub_net_2/dense_13/MatMul/ReadVariableOp7^feed_forward_sub_net_2/dense_14/BiasAdd/ReadVariableOp6^feed_forward_sub_net_2/dense_14/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2�
Afeed_forward_sub_net_2/batch_normalization_12/Cast/ReadVariableOpAfeed_forward_sub_net_2/batch_normalization_12/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_12/Cast_1/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_12/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_12/Cast_2/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_12/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_12/Cast_3/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_12/Cast_3/ReadVariableOp2�
Afeed_forward_sub_net_2/batch_normalization_13/Cast/ReadVariableOpAfeed_forward_sub_net_2/batch_normalization_13/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_13/Cast_1/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_13/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_13/Cast_2/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_13/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_13/Cast_3/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_13/Cast_3/ReadVariableOp2�
Afeed_forward_sub_net_2/batch_normalization_14/Cast/ReadVariableOpAfeed_forward_sub_net_2/batch_normalization_14/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_14/Cast_1/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_14/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_14/Cast_2/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_14/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_14/Cast_3/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_14/Cast_3/ReadVariableOp2�
Afeed_forward_sub_net_2/batch_normalization_15/Cast/ReadVariableOpAfeed_forward_sub_net_2/batch_normalization_15/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_15/Cast_1/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_15/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_15/Cast_2/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_15/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_15/Cast_3/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_15/Cast_3/ReadVariableOp2�
Afeed_forward_sub_net_2/batch_normalization_16/Cast/ReadVariableOpAfeed_forward_sub_net_2/batch_normalization_16/Cast/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_16/Cast_1/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_16/Cast_1/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_16/Cast_2/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_16/Cast_2/ReadVariableOp2�
Cfeed_forward_sub_net_2/batch_normalization_16/Cast_3/ReadVariableOpCfeed_forward_sub_net_2/batch_normalization_16/Cast_3/ReadVariableOp2n
5feed_forward_sub_net_2/dense_10/MatMul/ReadVariableOp5feed_forward_sub_net_2/dense_10/MatMul/ReadVariableOp2n
5feed_forward_sub_net_2/dense_11/MatMul/ReadVariableOp5feed_forward_sub_net_2/dense_11/MatMul/ReadVariableOp2n
5feed_forward_sub_net_2/dense_12/MatMul/ReadVariableOp5feed_forward_sub_net_2/dense_12/MatMul/ReadVariableOp2n
5feed_forward_sub_net_2/dense_13/MatMul/ReadVariableOp5feed_forward_sub_net_2/dense_13/MatMul/ReadVariableOp2p
6feed_forward_sub_net_2/dense_14/BiasAdd/ReadVariableOp6feed_forward_sub_net_2/dense_14/BiasAdd/ReadVariableOp2n
5feed_forward_sub_net_2/dense_14/MatMul/ReadVariableOp5feed_forward_sub_net_2/dense_14/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
��
�
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_315271
input_1A
3batch_normalization_12_cast_readvariableop_resource:
C
5batch_normalization_12_cast_1_readvariableop_resource:
C
5batch_normalization_12_cast_2_readvariableop_resource:
C
5batch_normalization_12_cast_3_readvariableop_resource:
9
'dense_10_matmul_readvariableop_resource:
A
3batch_normalization_13_cast_readvariableop_resource:C
5batch_normalization_13_cast_1_readvariableop_resource:C
5batch_normalization_13_cast_2_readvariableop_resource:C
5batch_normalization_13_cast_3_readvariableop_resource:9
'dense_11_matmul_readvariableop_resource:A
3batch_normalization_14_cast_readvariableop_resource:C
5batch_normalization_14_cast_1_readvariableop_resource:C
5batch_normalization_14_cast_2_readvariableop_resource:C
5batch_normalization_14_cast_3_readvariableop_resource:9
'dense_12_matmul_readvariableop_resource:A
3batch_normalization_15_cast_readvariableop_resource:C
5batch_normalization_15_cast_1_readvariableop_resource:C
5batch_normalization_15_cast_2_readvariableop_resource:C
5batch_normalization_15_cast_3_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:A
3batch_normalization_16_cast_readvariableop_resource:C
5batch_normalization_16_cast_1_readvariableop_resource:C
5batch_normalization_16_cast_2_readvariableop_resource:C
5batch_normalization_16_cast_3_readvariableop_resource:9
'dense_14_matmul_readvariableop_resource:
6
(dense_14_biasadd_readvariableop_resource:

identity��*batch_normalization_12/Cast/ReadVariableOp�,batch_normalization_12/Cast_1/ReadVariableOp�,batch_normalization_12/Cast_2/ReadVariableOp�,batch_normalization_12/Cast_3/ReadVariableOp�*batch_normalization_13/Cast/ReadVariableOp�,batch_normalization_13/Cast_1/ReadVariableOp�,batch_normalization_13/Cast_2/ReadVariableOp�,batch_normalization_13/Cast_3/ReadVariableOp�*batch_normalization_14/Cast/ReadVariableOp�,batch_normalization_14/Cast_1/ReadVariableOp�,batch_normalization_14/Cast_2/ReadVariableOp�,batch_normalization_14/Cast_3/ReadVariableOp�*batch_normalization_15/Cast/ReadVariableOp�,batch_normalization_15/Cast_1/ReadVariableOp�,batch_normalization_15/Cast_2/ReadVariableOp�,batch_normalization_15/Cast_3/ReadVariableOp�*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�,batch_normalization_16/Cast_2/ReadVariableOp�,batch_normalization_16/Cast_3/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
*batch_normalization_12/Cast/ReadVariableOpReadVariableOp3batch_normalization_12_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_12/Cast/ReadVariableOp�
,batch_normalization_12/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_12_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_12/Cast_1/ReadVariableOp�
,batch_normalization_12/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_12_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_12/Cast_2/ReadVariableOp�
,batch_normalization_12/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_12_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_12/Cast_3/ReadVariableOp�
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_12/batchnorm/add/y�
$batch_normalization_12/batchnorm/addAddV24batch_normalization_12/Cast_1/ReadVariableOp:value:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_12/batchnorm/add�
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_12/batchnorm/Rsqrt�
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:04batch_normalization_12/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_12/batchnorm/mul�
&batch_normalization_12/batchnorm/mul_1Mulinput_1(batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_12/batchnorm/mul_1�
&batch_normalization_12/batchnorm/mul_2Mul2batch_normalization_12/Cast/ReadVariableOp:value:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_12/batchnorm/mul_2�
$batch_normalization_12/batchnorm/subSub4batch_normalization_12/Cast_2/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_12/batchnorm/sub�
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_12/batchnorm/add_1�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMul*batch_normalization_12/batchnorm/add_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_10/MatMul�
*batch_normalization_13/Cast/ReadVariableOpReadVariableOp3batch_normalization_13_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_13/Cast/ReadVariableOp�
,batch_normalization_13/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_13_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_13/Cast_1/ReadVariableOp�
,batch_normalization_13/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_13_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_13/Cast_2/ReadVariableOp�
,batch_normalization_13/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_13_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_13/Cast_3/ReadVariableOp�
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_13/batchnorm/add/y�
$batch_normalization_13/batchnorm/addAddV24batch_normalization_13/Cast_1/ReadVariableOp:value:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/add�
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/Rsqrt�
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:04batch_normalization_13/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/mul�
&batch_normalization_13/batchnorm/mul_1Muldense_10/MatMul:product:0(batch_normalization_13/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_13/batchnorm/mul_1�
&batch_normalization_13/batchnorm/mul_2Mul2batch_normalization_13/Cast/ReadVariableOp:value:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/mul_2�
$batch_normalization_13/batchnorm/subSub4batch_normalization_13/Cast_2/ReadVariableOp:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/sub�
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_13/batchnorm/add_1r
ReluRelu*batch_normalization_13/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMulRelu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
*batch_normalization_14/Cast/ReadVariableOpReadVariableOp3batch_normalization_14_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_14/Cast/ReadVariableOp�
,batch_normalization_14/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_14_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_14/Cast_1/ReadVariableOp�
,batch_normalization_14/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_14_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_14/Cast_2/ReadVariableOp�
,batch_normalization_14/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_14_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_14/Cast_3/ReadVariableOp�
&batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_14/batchnorm/add/y�
$batch_normalization_14/batchnorm/addAddV24batch_normalization_14/Cast_1/ReadVariableOp:value:0/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/add�
&batch_normalization_14/batchnorm/RsqrtRsqrt(batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/Rsqrt�
$batch_normalization_14/batchnorm/mulMul*batch_normalization_14/batchnorm/Rsqrt:y:04batch_normalization_14/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/mul�
&batch_normalization_14/batchnorm/mul_1Muldense_11/MatMul:product:0(batch_normalization_14/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_14/batchnorm/mul_1�
&batch_normalization_14/batchnorm/mul_2Mul2batch_normalization_14/Cast/ReadVariableOp:value:0(batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/mul_2�
$batch_normalization_14/batchnorm/subSub4batch_normalization_14/Cast_2/ReadVariableOp:value:0*batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/sub�
&batch_normalization_14/batchnorm/add_1AddV2*batch_normalization_14/batchnorm/mul_1:z:0(batch_normalization_14/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_14/batchnorm/add_1v
Relu_1Relu*batch_normalization_14/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulRelu_1:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_12/MatMul�
*batch_normalization_15/Cast/ReadVariableOpReadVariableOp3batch_normalization_15_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_15/Cast/ReadVariableOp�
,batch_normalization_15/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_15_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_15/Cast_1/ReadVariableOp�
,batch_normalization_15/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_15_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_15/Cast_2/ReadVariableOp�
,batch_normalization_15/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_15_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_15/Cast_3/ReadVariableOp�
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_15/batchnorm/add/y�
$batch_normalization_15/batchnorm/addAddV24batch_normalization_15/Cast_1/ReadVariableOp:value:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/add�
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/Rsqrt�
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:04batch_normalization_15/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/mul�
&batch_normalization_15/batchnorm/mul_1Muldense_12/MatMul:product:0(batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_15/batchnorm/mul_1�
&batch_normalization_15/batchnorm/mul_2Mul2batch_normalization_15/Cast/ReadVariableOp:value:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/mul_2�
$batch_normalization_15/batchnorm/subSub4batch_normalization_15/Cast_2/ReadVariableOp:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/sub�
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_15/batchnorm/add_1v
Relu_2Relu*batch_normalization_15/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulRelu_2:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_13/MatMul�
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_16/Cast/ReadVariableOp�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_16/Cast_1/ReadVariableOp�
,batch_normalization_16/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_16_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_16/Cast_2/ReadVariableOp�
,batch_normalization_16/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_16_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_16/Cast_3/ReadVariableOp�
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_16/batchnorm/add/y�
$batch_normalization_16/batchnorm/addAddV24batch_normalization_16/Cast_1/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/add�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/Rsqrt�
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:04batch_normalization_16/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/mul�
&batch_normalization_16/batchnorm/mul_1Muldense_13/MatMul:product:0(batch_normalization_16/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_16/batchnorm/mul_1�
&batch_normalization_16/batchnorm/mul_2Mul2batch_normalization_16/Cast/ReadVariableOp:value:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/mul_2�
$batch_normalization_16/batchnorm/subSub4batch_normalization_16/Cast_2/ReadVariableOp:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/sub�
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_16/batchnorm/add_1v
Relu_3Relu*batch_normalization_16/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMulRelu_3:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_14/BiasAddt
IdentityIdentitydense_14/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�	
NoOpNoOp+^batch_normalization_12/Cast/ReadVariableOp-^batch_normalization_12/Cast_1/ReadVariableOp-^batch_normalization_12/Cast_2/ReadVariableOp-^batch_normalization_12/Cast_3/ReadVariableOp+^batch_normalization_13/Cast/ReadVariableOp-^batch_normalization_13/Cast_1/ReadVariableOp-^batch_normalization_13/Cast_2/ReadVariableOp-^batch_normalization_13/Cast_3/ReadVariableOp+^batch_normalization_14/Cast/ReadVariableOp-^batch_normalization_14/Cast_1/ReadVariableOp-^batch_normalization_14/Cast_2/ReadVariableOp-^batch_normalization_14/Cast_3/ReadVariableOp+^batch_normalization_15/Cast/ReadVariableOp-^batch_normalization_15/Cast_1/ReadVariableOp-^batch_normalization_15/Cast_2/ReadVariableOp-^batch_normalization_15/Cast_3/ReadVariableOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp-^batch_normalization_16/Cast_2/ReadVariableOp-^batch_normalization_16/Cast_3/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_12/MatMul/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_12/Cast/ReadVariableOp*batch_normalization_12/Cast/ReadVariableOp2\
,batch_normalization_12/Cast_1/ReadVariableOp,batch_normalization_12/Cast_1/ReadVariableOp2\
,batch_normalization_12/Cast_2/ReadVariableOp,batch_normalization_12/Cast_2/ReadVariableOp2\
,batch_normalization_12/Cast_3/ReadVariableOp,batch_normalization_12/Cast_3/ReadVariableOp2X
*batch_normalization_13/Cast/ReadVariableOp*batch_normalization_13/Cast/ReadVariableOp2\
,batch_normalization_13/Cast_1/ReadVariableOp,batch_normalization_13/Cast_1/ReadVariableOp2\
,batch_normalization_13/Cast_2/ReadVariableOp,batch_normalization_13/Cast_2/ReadVariableOp2\
,batch_normalization_13/Cast_3/ReadVariableOp,batch_normalization_13/Cast_3/ReadVariableOp2X
*batch_normalization_14/Cast/ReadVariableOp*batch_normalization_14/Cast/ReadVariableOp2\
,batch_normalization_14/Cast_1/ReadVariableOp,batch_normalization_14/Cast_1/ReadVariableOp2\
,batch_normalization_14/Cast_2/ReadVariableOp,batch_normalization_14/Cast_2/ReadVariableOp2\
,batch_normalization_14/Cast_3/ReadVariableOp,batch_normalization_14/Cast_3/ReadVariableOp2X
*batch_normalization_15/Cast/ReadVariableOp*batch_normalization_15/Cast/ReadVariableOp2\
,batch_normalization_15/Cast_1/ReadVariableOp,batch_normalization_15/Cast_1/ReadVariableOp2\
,batch_normalization_15/Cast_2/ReadVariableOp,batch_normalization_15/Cast_2/ReadVariableOp2\
,batch_normalization_15/Cast_3/ReadVariableOp,batch_normalization_15/Cast_3/ReadVariableOp2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2\
,batch_normalization_16/Cast_2/ReadVariableOp,batch_normalization_16/Cast_2/ReadVariableOp2\
,batch_normalization_16/Cast_3/ReadVariableOp,batch_normalization_16/Cast_3/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
�z
�
"__inference__traced_restore_316131
file_prefixR
Dassignvariableop_feed_forward_sub_net_2_batch_normalization_12_gamma:
S
Eassignvariableop_1_feed_forward_sub_net_2_batch_normalization_12_beta:
T
Fassignvariableop_2_feed_forward_sub_net_2_batch_normalization_13_gamma:S
Eassignvariableop_3_feed_forward_sub_net_2_batch_normalization_13_beta:T
Fassignvariableop_4_feed_forward_sub_net_2_batch_normalization_14_gamma:S
Eassignvariableop_5_feed_forward_sub_net_2_batch_normalization_14_beta:T
Fassignvariableop_6_feed_forward_sub_net_2_batch_normalization_15_gamma:S
Eassignvariableop_7_feed_forward_sub_net_2_batch_normalization_15_beta:T
Fassignvariableop_8_feed_forward_sub_net_2_batch_normalization_16_gamma:S
Eassignvariableop_9_feed_forward_sub_net_2_batch_normalization_16_beta:[
Massignvariableop_10_feed_forward_sub_net_2_batch_normalization_12_moving_mean:
_
Qassignvariableop_11_feed_forward_sub_net_2_batch_normalization_12_moving_variance:
[
Massignvariableop_12_feed_forward_sub_net_2_batch_normalization_13_moving_mean:_
Qassignvariableop_13_feed_forward_sub_net_2_batch_normalization_13_moving_variance:[
Massignvariableop_14_feed_forward_sub_net_2_batch_normalization_14_moving_mean:_
Qassignvariableop_15_feed_forward_sub_net_2_batch_normalization_14_moving_variance:[
Massignvariableop_16_feed_forward_sub_net_2_batch_normalization_15_moving_mean:_
Qassignvariableop_17_feed_forward_sub_net_2_batch_normalization_15_moving_variance:[
Massignvariableop_18_feed_forward_sub_net_2_batch_normalization_16_moving_mean:_
Qassignvariableop_19_feed_forward_sub_net_2_batch_normalization_16_moving_variance:L
:assignvariableop_20_feed_forward_sub_net_2_dense_10_kernel:
L
:assignvariableop_21_feed_forward_sub_net_2_dense_11_kernel:L
:assignvariableop_22_feed_forward_sub_net_2_dense_12_kernel:L
:assignvariableop_23_feed_forward_sub_net_2_dense_13_kernel:L
:assignvariableop_24_feed_forward_sub_net_2_dense_14_kernel:
F
8assignvariableop_25_feed_forward_sub_net_2_dense_14_bias:
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
AssignVariableOpAssignVariableOpDassignvariableop_feed_forward_sub_net_2_batch_normalization_12_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpEassignvariableop_1_feed_forward_sub_net_2_batch_normalization_12_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpFassignvariableop_2_feed_forward_sub_net_2_batch_normalization_13_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpEassignvariableop_3_feed_forward_sub_net_2_batch_normalization_13_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpFassignvariableop_4_feed_forward_sub_net_2_batch_normalization_14_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpEassignvariableop_5_feed_forward_sub_net_2_batch_normalization_14_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpFassignvariableop_6_feed_forward_sub_net_2_batch_normalization_15_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpEassignvariableop_7_feed_forward_sub_net_2_batch_normalization_15_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpFassignvariableop_8_feed_forward_sub_net_2_batch_normalization_16_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpEassignvariableop_9_feed_forward_sub_net_2_batch_normalization_16_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpMassignvariableop_10_feed_forward_sub_net_2_batch_normalization_12_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpQassignvariableop_11_feed_forward_sub_net_2_batch_normalization_12_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpMassignvariableop_12_feed_forward_sub_net_2_batch_normalization_13_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpQassignvariableop_13_feed_forward_sub_net_2_batch_normalization_13_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpMassignvariableop_14_feed_forward_sub_net_2_batch_normalization_14_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpQassignvariableop_15_feed_forward_sub_net_2_batch_normalization_14_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpMassignvariableop_16_feed_forward_sub_net_2_batch_normalization_15_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpQassignvariableop_17_feed_forward_sub_net_2_batch_normalization_15_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpMassignvariableop_18_feed_forward_sub_net_2_batch_normalization_16_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpQassignvariableop_19_feed_forward_sub_net_2_batch_normalization_16_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp:assignvariableop_20_feed_forward_sub_net_2_dense_10_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp:assignvariableop_21_feed_forward_sub_net_2_dense_11_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp:assignvariableop_22_feed_forward_sub_net_2_dense_12_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp:assignvariableop_23_feed_forward_sub_net_2_dense_13_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp:assignvariableop_24_feed_forward_sub_net_2_dense_14_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp8assignvariableop_25_feed_forward_sub_net_2_dense_14_biasIdentity_25:output:0"/device:CPU:0*
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
�+
�
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_313247

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
�
�
7__inference_feed_forward_sub_net_2_layer_call_fn_314702
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
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_3141082
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
D__inference_dense_12_layer_call_and_return_conditional_losses_315909

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
D__inference_dense_12_layer_call_and_return_conditional_losses_314056

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
$__inference_signature_wrapper_314645
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
!__inference__wrapped_model_3131612
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
7__inference_batch_normalization_16_layer_call_fn_315811

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
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3139112
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
��
�
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_314979
xA
3batch_normalization_12_cast_readvariableop_resource:
C
5batch_normalization_12_cast_1_readvariableop_resource:
C
5batch_normalization_12_cast_2_readvariableop_resource:
C
5batch_normalization_12_cast_3_readvariableop_resource:
9
'dense_10_matmul_readvariableop_resource:
A
3batch_normalization_13_cast_readvariableop_resource:C
5batch_normalization_13_cast_1_readvariableop_resource:C
5batch_normalization_13_cast_2_readvariableop_resource:C
5batch_normalization_13_cast_3_readvariableop_resource:9
'dense_11_matmul_readvariableop_resource:A
3batch_normalization_14_cast_readvariableop_resource:C
5batch_normalization_14_cast_1_readvariableop_resource:C
5batch_normalization_14_cast_2_readvariableop_resource:C
5batch_normalization_14_cast_3_readvariableop_resource:9
'dense_12_matmul_readvariableop_resource:A
3batch_normalization_15_cast_readvariableop_resource:C
5batch_normalization_15_cast_1_readvariableop_resource:C
5batch_normalization_15_cast_2_readvariableop_resource:C
5batch_normalization_15_cast_3_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:A
3batch_normalization_16_cast_readvariableop_resource:C
5batch_normalization_16_cast_1_readvariableop_resource:C
5batch_normalization_16_cast_2_readvariableop_resource:C
5batch_normalization_16_cast_3_readvariableop_resource:9
'dense_14_matmul_readvariableop_resource:
6
(dense_14_biasadd_readvariableop_resource:

identity��*batch_normalization_12/Cast/ReadVariableOp�,batch_normalization_12/Cast_1/ReadVariableOp�,batch_normalization_12/Cast_2/ReadVariableOp�,batch_normalization_12/Cast_3/ReadVariableOp�*batch_normalization_13/Cast/ReadVariableOp�,batch_normalization_13/Cast_1/ReadVariableOp�,batch_normalization_13/Cast_2/ReadVariableOp�,batch_normalization_13/Cast_3/ReadVariableOp�*batch_normalization_14/Cast/ReadVariableOp�,batch_normalization_14/Cast_1/ReadVariableOp�,batch_normalization_14/Cast_2/ReadVariableOp�,batch_normalization_14/Cast_3/ReadVariableOp�*batch_normalization_15/Cast/ReadVariableOp�,batch_normalization_15/Cast_1/ReadVariableOp�,batch_normalization_15/Cast_2/ReadVariableOp�,batch_normalization_15/Cast_3/ReadVariableOp�*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�,batch_normalization_16/Cast_2/ReadVariableOp�,batch_normalization_16/Cast_3/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
*batch_normalization_12/Cast/ReadVariableOpReadVariableOp3batch_normalization_12_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_12/Cast/ReadVariableOp�
,batch_normalization_12/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_12_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_12/Cast_1/ReadVariableOp�
,batch_normalization_12/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_12_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_12/Cast_2/ReadVariableOp�
,batch_normalization_12/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_12_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_12/Cast_3/ReadVariableOp�
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_12/batchnorm/add/y�
$batch_normalization_12/batchnorm/addAddV24batch_normalization_12/Cast_1/ReadVariableOp:value:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_12/batchnorm/add�
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_12/batchnorm/Rsqrt�
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:04batch_normalization_12/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_12/batchnorm/mul�
&batch_normalization_12/batchnorm/mul_1Mulx(batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_12/batchnorm/mul_1�
&batch_normalization_12/batchnorm/mul_2Mul2batch_normalization_12/Cast/ReadVariableOp:value:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_12/batchnorm/mul_2�
$batch_normalization_12/batchnorm/subSub4batch_normalization_12/Cast_2/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_12/batchnorm/sub�
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_12/batchnorm/add_1�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMul*batch_normalization_12/batchnorm/add_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_10/MatMul�
*batch_normalization_13/Cast/ReadVariableOpReadVariableOp3batch_normalization_13_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_13/Cast/ReadVariableOp�
,batch_normalization_13/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_13_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_13/Cast_1/ReadVariableOp�
,batch_normalization_13/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_13_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_13/Cast_2/ReadVariableOp�
,batch_normalization_13/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_13_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_13/Cast_3/ReadVariableOp�
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_13/batchnorm/add/y�
$batch_normalization_13/batchnorm/addAddV24batch_normalization_13/Cast_1/ReadVariableOp:value:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/add�
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/Rsqrt�
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:04batch_normalization_13/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/mul�
&batch_normalization_13/batchnorm/mul_1Muldense_10/MatMul:product:0(batch_normalization_13/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_13/batchnorm/mul_1�
&batch_normalization_13/batchnorm/mul_2Mul2batch_normalization_13/Cast/ReadVariableOp:value:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/mul_2�
$batch_normalization_13/batchnorm/subSub4batch_normalization_13/Cast_2/ReadVariableOp:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/sub�
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_13/batchnorm/add_1r
ReluRelu*batch_normalization_13/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMulRelu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
*batch_normalization_14/Cast/ReadVariableOpReadVariableOp3batch_normalization_14_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_14/Cast/ReadVariableOp�
,batch_normalization_14/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_14_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_14/Cast_1/ReadVariableOp�
,batch_normalization_14/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_14_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_14/Cast_2/ReadVariableOp�
,batch_normalization_14/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_14_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_14/Cast_3/ReadVariableOp�
&batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_14/batchnorm/add/y�
$batch_normalization_14/batchnorm/addAddV24batch_normalization_14/Cast_1/ReadVariableOp:value:0/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/add�
&batch_normalization_14/batchnorm/RsqrtRsqrt(batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/Rsqrt�
$batch_normalization_14/batchnorm/mulMul*batch_normalization_14/batchnorm/Rsqrt:y:04batch_normalization_14/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/mul�
&batch_normalization_14/batchnorm/mul_1Muldense_11/MatMul:product:0(batch_normalization_14/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_14/batchnorm/mul_1�
&batch_normalization_14/batchnorm/mul_2Mul2batch_normalization_14/Cast/ReadVariableOp:value:0(batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/mul_2�
$batch_normalization_14/batchnorm/subSub4batch_normalization_14/Cast_2/ReadVariableOp:value:0*batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/sub�
&batch_normalization_14/batchnorm/add_1AddV2*batch_normalization_14/batchnorm/mul_1:z:0(batch_normalization_14/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_14/batchnorm/add_1v
Relu_1Relu*batch_normalization_14/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulRelu_1:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_12/MatMul�
*batch_normalization_15/Cast/ReadVariableOpReadVariableOp3batch_normalization_15_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_15/Cast/ReadVariableOp�
,batch_normalization_15/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_15_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_15/Cast_1/ReadVariableOp�
,batch_normalization_15/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_15_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_15/Cast_2/ReadVariableOp�
,batch_normalization_15/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_15_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_15/Cast_3/ReadVariableOp�
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_15/batchnorm/add/y�
$batch_normalization_15/batchnorm/addAddV24batch_normalization_15/Cast_1/ReadVariableOp:value:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/add�
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/Rsqrt�
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:04batch_normalization_15/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/mul�
&batch_normalization_15/batchnorm/mul_1Muldense_12/MatMul:product:0(batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_15/batchnorm/mul_1�
&batch_normalization_15/batchnorm/mul_2Mul2batch_normalization_15/Cast/ReadVariableOp:value:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/mul_2�
$batch_normalization_15/batchnorm/subSub4batch_normalization_15/Cast_2/ReadVariableOp:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/sub�
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_15/batchnorm/add_1v
Relu_2Relu*batch_normalization_15/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulRelu_2:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_13/MatMul�
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_16/Cast/ReadVariableOp�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_16/Cast_1/ReadVariableOp�
,batch_normalization_16/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_16_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_16/Cast_2/ReadVariableOp�
,batch_normalization_16/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_16_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_16/Cast_3/ReadVariableOp�
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_16/batchnorm/add/y�
$batch_normalization_16/batchnorm/addAddV24batch_normalization_16/Cast_1/ReadVariableOp:value:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/add�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/Rsqrt�
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:04batch_normalization_16/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/mul�
&batch_normalization_16/batchnorm/mul_1Muldense_13/MatMul:product:0(batch_normalization_16/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_16/batchnorm/mul_1�
&batch_normalization_16/batchnorm/mul_2Mul2batch_normalization_16/Cast/ReadVariableOp:value:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/mul_2�
$batch_normalization_16/batchnorm/subSub4batch_normalization_16/Cast_2/ReadVariableOp:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/sub�
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_16/batchnorm/add_1v
Relu_3Relu*batch_normalization_16/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMulRelu_3:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_14/BiasAddt
IdentityIdentitydense_14/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�	
NoOpNoOp+^batch_normalization_12/Cast/ReadVariableOp-^batch_normalization_12/Cast_1/ReadVariableOp-^batch_normalization_12/Cast_2/ReadVariableOp-^batch_normalization_12/Cast_3/ReadVariableOp+^batch_normalization_13/Cast/ReadVariableOp-^batch_normalization_13/Cast_1/ReadVariableOp-^batch_normalization_13/Cast_2/ReadVariableOp-^batch_normalization_13/Cast_3/ReadVariableOp+^batch_normalization_14/Cast/ReadVariableOp-^batch_normalization_14/Cast_1/ReadVariableOp-^batch_normalization_14/Cast_2/ReadVariableOp-^batch_normalization_14/Cast_3/ReadVariableOp+^batch_normalization_15/Cast/ReadVariableOp-^batch_normalization_15/Cast_1/ReadVariableOp-^batch_normalization_15/Cast_2/ReadVariableOp-^batch_normalization_15/Cast_3/ReadVariableOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp-^batch_normalization_16/Cast_2/ReadVariableOp-^batch_normalization_16/Cast_3/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_12/MatMul/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_12/Cast/ReadVariableOp*batch_normalization_12/Cast/ReadVariableOp2\
,batch_normalization_12/Cast_1/ReadVariableOp,batch_normalization_12/Cast_1/ReadVariableOp2\
,batch_normalization_12/Cast_2/ReadVariableOp,batch_normalization_12/Cast_2/ReadVariableOp2\
,batch_normalization_12/Cast_3/ReadVariableOp,batch_normalization_12/Cast_3/ReadVariableOp2X
*batch_normalization_13/Cast/ReadVariableOp*batch_normalization_13/Cast/ReadVariableOp2\
,batch_normalization_13/Cast_1/ReadVariableOp,batch_normalization_13/Cast_1/ReadVariableOp2\
,batch_normalization_13/Cast_2/ReadVariableOp,batch_normalization_13/Cast_2/ReadVariableOp2\
,batch_normalization_13/Cast_3/ReadVariableOp,batch_normalization_13/Cast_3/ReadVariableOp2X
*batch_normalization_14/Cast/ReadVariableOp*batch_normalization_14/Cast/ReadVariableOp2\
,batch_normalization_14/Cast_1/ReadVariableOp,batch_normalization_14/Cast_1/ReadVariableOp2\
,batch_normalization_14/Cast_2/ReadVariableOp,batch_normalization_14/Cast_2/ReadVariableOp2\
,batch_normalization_14/Cast_3/ReadVariableOp,batch_normalization_14/Cast_3/ReadVariableOp2X
*batch_normalization_15/Cast/ReadVariableOp*batch_normalization_15/Cast/ReadVariableOp2\
,batch_normalization_15/Cast_1/ReadVariableOp,batch_normalization_15/Cast_1/ReadVariableOp2\
,batch_normalization_15/Cast_2/ReadVariableOp,batch_normalization_15/Cast_2/ReadVariableOp2\
,batch_normalization_15/Cast_3/ReadVariableOp,batch_normalization_15/Cast_3/ReadVariableOp2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2\
,batch_normalization_16/Cast_2/ReadVariableOp,batch_normalization_16/Cast_2/ReadVariableOp2\
,batch_normalization_16/Cast_3/ReadVariableOp,batch_normalization_16/Cast_3/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������


_user_specified_namex
�
�
D__inference_dense_13_layer_call_and_return_conditional_losses_314077

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
D__inference_dense_10_layer_call_and_return_conditional_losses_314014

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
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_315867

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
D__inference_dense_11_layer_call_and_return_conditional_losses_314035

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
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_315667

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
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_315621

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
�E
�
__inference__traced_save_316043
file_prefixR
Nsavev2_feed_forward_sub_net_2_batch_normalization_12_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_2_batch_normalization_12_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_2_batch_normalization_13_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_2_batch_normalization_13_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_2_batch_normalization_14_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_2_batch_normalization_14_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_2_batch_normalization_15_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_2_batch_normalization_15_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_2_batch_normalization_16_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_2_batch_normalization_16_beta_read_readvariableopX
Tsavev2_feed_forward_sub_net_2_batch_normalization_12_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_2_batch_normalization_12_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_2_batch_normalization_13_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_2_batch_normalization_13_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_2_batch_normalization_14_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_2_batch_normalization_14_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_2_batch_normalization_15_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_2_batch_normalization_15_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_2_batch_normalization_16_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_2_batch_normalization_16_moving_variance_read_readvariableopE
Asavev2_feed_forward_sub_net_2_dense_10_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_2_dense_11_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_2_dense_12_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_2_dense_13_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_2_dense_14_kernel_read_readvariableopC
?savev2_feed_forward_sub_net_2_dense_14_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Nsavev2_feed_forward_sub_net_2_batch_normalization_12_gamma_read_readvariableopMsavev2_feed_forward_sub_net_2_batch_normalization_12_beta_read_readvariableopNsavev2_feed_forward_sub_net_2_batch_normalization_13_gamma_read_readvariableopMsavev2_feed_forward_sub_net_2_batch_normalization_13_beta_read_readvariableopNsavev2_feed_forward_sub_net_2_batch_normalization_14_gamma_read_readvariableopMsavev2_feed_forward_sub_net_2_batch_normalization_14_beta_read_readvariableopNsavev2_feed_forward_sub_net_2_batch_normalization_15_gamma_read_readvariableopMsavev2_feed_forward_sub_net_2_batch_normalization_15_beta_read_readvariableopNsavev2_feed_forward_sub_net_2_batch_normalization_16_gamma_read_readvariableopMsavev2_feed_forward_sub_net_2_batch_normalization_16_beta_read_readvariableopTsavev2_feed_forward_sub_net_2_batch_normalization_12_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_2_batch_normalization_12_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_2_batch_normalization_13_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_2_batch_normalization_13_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_2_batch_normalization_14_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_2_batch_normalization_14_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_2_batch_normalization_15_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_2_batch_normalization_15_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_2_batch_normalization_16_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_2_batch_normalization_16_moving_variance_read_readvariableopAsavev2_feed_forward_sub_net_2_dense_10_kernel_read_readvariableopAsavev2_feed_forward_sub_net_2_dense_11_kernel_read_readvariableopAsavev2_feed_forward_sub_net_2_dense_12_kernel_read_readvariableopAsavev2_feed_forward_sub_net_2_dense_13_kernel_read_readvariableopAsavev2_feed_forward_sub_net_2_dense_14_kernel_read_readvariableop?savev2_feed_forward_sub_net_2_dense_14_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�
�
7__inference_batch_normalization_13_layer_call_fn_315552

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
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3133512
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
7__inference_batch_normalization_12_layer_call_fn_315483

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
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3132472
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
�

�
D__inference_dense_14_layer_call_and_return_conditional_losses_314101

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
�+
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_313911

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
7__inference_batch_normalization_14_layer_call_fn_315634

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
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_3135172
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
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_315831

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
7__inference_batch_normalization_16_layer_call_fn_315798

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
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_3138492
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
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_313745

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
��
�
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_315457
input_1L
>batch_normalization_12_assignmovingavg_readvariableop_resource:
N
@batch_normalization_12_assignmovingavg_1_readvariableop_resource:
A
3batch_normalization_12_cast_readvariableop_resource:
C
5batch_normalization_12_cast_1_readvariableop_resource:
9
'dense_10_matmul_readvariableop_resource:
L
>batch_normalization_13_assignmovingavg_readvariableop_resource:N
@batch_normalization_13_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_13_cast_readvariableop_resource:C
5batch_normalization_13_cast_1_readvariableop_resource:9
'dense_11_matmul_readvariableop_resource:L
>batch_normalization_14_assignmovingavg_readvariableop_resource:N
@batch_normalization_14_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_14_cast_readvariableop_resource:C
5batch_normalization_14_cast_1_readvariableop_resource:9
'dense_12_matmul_readvariableop_resource:L
>batch_normalization_15_assignmovingavg_readvariableop_resource:N
@batch_normalization_15_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_15_cast_readvariableop_resource:C
5batch_normalization_15_cast_1_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:L
>batch_normalization_16_assignmovingavg_readvariableop_resource:N
@batch_normalization_16_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_16_cast_readvariableop_resource:C
5batch_normalization_16_cast_1_readvariableop_resource:9
'dense_14_matmul_readvariableop_resource:
6
(dense_14_biasadd_readvariableop_resource:

identity��&batch_normalization_12/AssignMovingAvg�5batch_normalization_12/AssignMovingAvg/ReadVariableOp�(batch_normalization_12/AssignMovingAvg_1�7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_12/Cast/ReadVariableOp�,batch_normalization_12/Cast_1/ReadVariableOp�&batch_normalization_13/AssignMovingAvg�5batch_normalization_13/AssignMovingAvg/ReadVariableOp�(batch_normalization_13/AssignMovingAvg_1�7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_13/Cast/ReadVariableOp�,batch_normalization_13/Cast_1/ReadVariableOp�&batch_normalization_14/AssignMovingAvg�5batch_normalization_14/AssignMovingAvg/ReadVariableOp�(batch_normalization_14/AssignMovingAvg_1�7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_14/Cast/ReadVariableOp�,batch_normalization_14/Cast_1/ReadVariableOp�&batch_normalization_15/AssignMovingAvg�5batch_normalization_15/AssignMovingAvg/ReadVariableOp�(batch_normalization_15/AssignMovingAvg_1�7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_15/Cast/ReadVariableOp�,batch_normalization_15/Cast_1/ReadVariableOp�&batch_normalization_16/AssignMovingAvg�5batch_normalization_16/AssignMovingAvg/ReadVariableOp�(batch_normalization_16/AssignMovingAvg_1�7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
5batch_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_12/moments/mean/reduction_indices�
#batch_normalization_12/moments/meanMeaninput_1>batch_normalization_12/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_12/moments/mean�
+batch_normalization_12/moments/StopGradientStopGradient,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_12/moments/StopGradient�
0batch_normalization_12/moments/SquaredDifferenceSquaredDifferenceinput_14batch_normalization_12/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������
22
0batch_normalization_12/moments/SquaredDifference�
9batch_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_12/moments/variance/reduction_indices�
'batch_normalization_12/moments/varianceMean4batch_normalization_12/moments/SquaredDifference:z:0Bbatch_normalization_12/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_12/moments/variance�
&batch_normalization_12/moments/SqueezeSqueeze,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_12/moments/Squeeze�
(batch_normalization_12/moments/Squeeze_1Squeeze0batch_normalization_12/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_12/moments/Squeeze_1�
,batch_normalization_12/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_12/AssignMovingAvg/decay�
+batch_normalization_12/AssignMovingAvg/CastCast5batch_normalization_12/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_12/AssignMovingAvg/Cast�
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_12_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_12/AssignMovingAvg/ReadVariableOp�
*batch_normalization_12/AssignMovingAvg/subSub=batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_12/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_12/AssignMovingAvg/sub�
*batch_normalization_12/AssignMovingAvg/mulMul.batch_normalization_12/AssignMovingAvg/sub:z:0/batch_normalization_12/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_12/AssignMovingAvg/mul�
&batch_normalization_12/AssignMovingAvgAssignSubVariableOp>batch_normalization_12_assignmovingavg_readvariableop_resource.batch_normalization_12/AssignMovingAvg/mul:z:06^batch_normalization_12/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_12/AssignMovingAvg�
.batch_normalization_12/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_12/AssignMovingAvg_1/decay�
-batch_normalization_12/AssignMovingAvg_1/CastCast7batch_normalization_12/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_12/AssignMovingAvg_1/Cast�
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_12_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_12/AssignMovingAvg_1/subSub?batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_12/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_12/AssignMovingAvg_1/sub�
,batch_normalization_12/AssignMovingAvg_1/mulMul0batch_normalization_12/AssignMovingAvg_1/sub:z:01batch_normalization_12/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_12/AssignMovingAvg_1/mul�
(batch_normalization_12/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_12_assignmovingavg_1_readvariableop_resource0batch_normalization_12/AssignMovingAvg_1/mul:z:08^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_12/AssignMovingAvg_1�
*batch_normalization_12/Cast/ReadVariableOpReadVariableOp3batch_normalization_12_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_12/Cast/ReadVariableOp�
,batch_normalization_12/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_12_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_12/Cast_1/ReadVariableOp�
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_12/batchnorm/add/y�
$batch_normalization_12/batchnorm/addAddV21batch_normalization_12/moments/Squeeze_1:output:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_12/batchnorm/add�
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_12/batchnorm/Rsqrt�
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:04batch_normalization_12/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_12/batchnorm/mul�
&batch_normalization_12/batchnorm/mul_1Mulinput_1(batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_12/batchnorm/mul_1�
&batch_normalization_12/batchnorm/mul_2Mul/batch_normalization_12/moments/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_12/batchnorm/mul_2�
$batch_normalization_12/batchnorm/subSub2batch_normalization_12/Cast/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_12/batchnorm/sub�
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_12/batchnorm/add_1�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMul*batch_normalization_12/batchnorm/add_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_10/MatMul�
5batch_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_13/moments/mean/reduction_indices�
#batch_normalization_13/moments/meanMeandense_10/MatMul:product:0>batch_normalization_13/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_13/moments/mean�
+batch_normalization_13/moments/StopGradientStopGradient,batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_13/moments/StopGradient�
0batch_normalization_13/moments/SquaredDifferenceSquaredDifferencedense_10/MatMul:product:04batch_normalization_13/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_13/moments/SquaredDifference�
9batch_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_13/moments/variance/reduction_indices�
'batch_normalization_13/moments/varianceMean4batch_normalization_13/moments/SquaredDifference:z:0Bbatch_normalization_13/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_13/moments/variance�
&batch_normalization_13/moments/SqueezeSqueeze,batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_13/moments/Squeeze�
(batch_normalization_13/moments/Squeeze_1Squeeze0batch_normalization_13/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_13/moments/Squeeze_1�
,batch_normalization_13/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_13/AssignMovingAvg/decay�
+batch_normalization_13/AssignMovingAvg/CastCast5batch_normalization_13/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_13/AssignMovingAvg/Cast�
5batch_normalization_13/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_13_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_13/AssignMovingAvg/ReadVariableOp�
*batch_normalization_13/AssignMovingAvg/subSub=batch_normalization_13/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_13/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_13/AssignMovingAvg/sub�
*batch_normalization_13/AssignMovingAvg/mulMul.batch_normalization_13/AssignMovingAvg/sub:z:0/batch_normalization_13/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_13/AssignMovingAvg/mul�
&batch_normalization_13/AssignMovingAvgAssignSubVariableOp>batch_normalization_13_assignmovingavg_readvariableop_resource.batch_normalization_13/AssignMovingAvg/mul:z:06^batch_normalization_13/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_13/AssignMovingAvg�
.batch_normalization_13/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_13/AssignMovingAvg_1/decay�
-batch_normalization_13/AssignMovingAvg_1/CastCast7batch_normalization_13/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_13/AssignMovingAvg_1/Cast�
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_13_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_13/AssignMovingAvg_1/subSub?batch_normalization_13/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_13/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_13/AssignMovingAvg_1/sub�
,batch_normalization_13/AssignMovingAvg_1/mulMul0batch_normalization_13/AssignMovingAvg_1/sub:z:01batch_normalization_13/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_13/AssignMovingAvg_1/mul�
(batch_normalization_13/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_13_assignmovingavg_1_readvariableop_resource0batch_normalization_13/AssignMovingAvg_1/mul:z:08^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_13/AssignMovingAvg_1�
*batch_normalization_13/Cast/ReadVariableOpReadVariableOp3batch_normalization_13_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_13/Cast/ReadVariableOp�
,batch_normalization_13/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_13_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_13/Cast_1/ReadVariableOp�
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_13/batchnorm/add/y�
$batch_normalization_13/batchnorm/addAddV21batch_normalization_13/moments/Squeeze_1:output:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/add�
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/Rsqrt�
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:04batch_normalization_13/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/mul�
&batch_normalization_13/batchnorm/mul_1Muldense_10/MatMul:product:0(batch_normalization_13/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_13/batchnorm/mul_1�
&batch_normalization_13/batchnorm/mul_2Mul/batch_normalization_13/moments/Squeeze:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/mul_2�
$batch_normalization_13/batchnorm/subSub2batch_normalization_13/Cast/ReadVariableOp:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/sub�
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_13/batchnorm/add_1r
ReluRelu*batch_normalization_13/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMulRelu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
5batch_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_14/moments/mean/reduction_indices�
#batch_normalization_14/moments/meanMeandense_11/MatMul:product:0>batch_normalization_14/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_14/moments/mean�
+batch_normalization_14/moments/StopGradientStopGradient,batch_normalization_14/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_14/moments/StopGradient�
0batch_normalization_14/moments/SquaredDifferenceSquaredDifferencedense_11/MatMul:product:04batch_normalization_14/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_14/moments/SquaredDifference�
9batch_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_14/moments/variance/reduction_indices�
'batch_normalization_14/moments/varianceMean4batch_normalization_14/moments/SquaredDifference:z:0Bbatch_normalization_14/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_14/moments/variance�
&batch_normalization_14/moments/SqueezeSqueeze,batch_normalization_14/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_14/moments/Squeeze�
(batch_normalization_14/moments/Squeeze_1Squeeze0batch_normalization_14/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_14/moments/Squeeze_1�
,batch_normalization_14/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_14/AssignMovingAvg/decay�
+batch_normalization_14/AssignMovingAvg/CastCast5batch_normalization_14/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_14/AssignMovingAvg/Cast�
5batch_normalization_14/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_14_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_14/AssignMovingAvg/ReadVariableOp�
*batch_normalization_14/AssignMovingAvg/subSub=batch_normalization_14/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_14/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_14/AssignMovingAvg/sub�
*batch_normalization_14/AssignMovingAvg/mulMul.batch_normalization_14/AssignMovingAvg/sub:z:0/batch_normalization_14/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_14/AssignMovingAvg/mul�
&batch_normalization_14/AssignMovingAvgAssignSubVariableOp>batch_normalization_14_assignmovingavg_readvariableop_resource.batch_normalization_14/AssignMovingAvg/mul:z:06^batch_normalization_14/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_14/AssignMovingAvg�
.batch_normalization_14/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_14/AssignMovingAvg_1/decay�
-batch_normalization_14/AssignMovingAvg_1/CastCast7batch_normalization_14/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_14/AssignMovingAvg_1/Cast�
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_14_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_14/AssignMovingAvg_1/subSub?batch_normalization_14/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_14/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_14/AssignMovingAvg_1/sub�
,batch_normalization_14/AssignMovingAvg_1/mulMul0batch_normalization_14/AssignMovingAvg_1/sub:z:01batch_normalization_14/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_14/AssignMovingAvg_1/mul�
(batch_normalization_14/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_14_assignmovingavg_1_readvariableop_resource0batch_normalization_14/AssignMovingAvg_1/mul:z:08^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_14/AssignMovingAvg_1�
*batch_normalization_14/Cast/ReadVariableOpReadVariableOp3batch_normalization_14_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_14/Cast/ReadVariableOp�
,batch_normalization_14/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_14_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_14/Cast_1/ReadVariableOp�
&batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_14/batchnorm/add/y�
$batch_normalization_14/batchnorm/addAddV21batch_normalization_14/moments/Squeeze_1:output:0/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/add�
&batch_normalization_14/batchnorm/RsqrtRsqrt(batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/Rsqrt�
$batch_normalization_14/batchnorm/mulMul*batch_normalization_14/batchnorm/Rsqrt:y:04batch_normalization_14/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/mul�
&batch_normalization_14/batchnorm/mul_1Muldense_11/MatMul:product:0(batch_normalization_14/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_14/batchnorm/mul_1�
&batch_normalization_14/batchnorm/mul_2Mul/batch_normalization_14/moments/Squeeze:output:0(batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/mul_2�
$batch_normalization_14/batchnorm/subSub2batch_normalization_14/Cast/ReadVariableOp:value:0*batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/sub�
&batch_normalization_14/batchnorm/add_1AddV2*batch_normalization_14/batchnorm/mul_1:z:0(batch_normalization_14/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_14/batchnorm/add_1v
Relu_1Relu*batch_normalization_14/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulRelu_1:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_12/MatMul�
5batch_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_15/moments/mean/reduction_indices�
#batch_normalization_15/moments/meanMeandense_12/MatMul:product:0>batch_normalization_15/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_15/moments/mean�
+batch_normalization_15/moments/StopGradientStopGradient,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_15/moments/StopGradient�
0batch_normalization_15/moments/SquaredDifferenceSquaredDifferencedense_12/MatMul:product:04batch_normalization_15/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_15/moments/SquaredDifference�
9batch_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_15/moments/variance/reduction_indices�
'batch_normalization_15/moments/varianceMean4batch_normalization_15/moments/SquaredDifference:z:0Bbatch_normalization_15/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_15/moments/variance�
&batch_normalization_15/moments/SqueezeSqueeze,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_15/moments/Squeeze�
(batch_normalization_15/moments/Squeeze_1Squeeze0batch_normalization_15/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_15/moments/Squeeze_1�
,batch_normalization_15/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_15/AssignMovingAvg/decay�
+batch_normalization_15/AssignMovingAvg/CastCast5batch_normalization_15/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_15/AssignMovingAvg/Cast�
5batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_15_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_15/AssignMovingAvg/ReadVariableOp�
*batch_normalization_15/AssignMovingAvg/subSub=batch_normalization_15/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_15/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_15/AssignMovingAvg/sub�
*batch_normalization_15/AssignMovingAvg/mulMul.batch_normalization_15/AssignMovingAvg/sub:z:0/batch_normalization_15/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_15/AssignMovingAvg/mul�
&batch_normalization_15/AssignMovingAvgAssignSubVariableOp>batch_normalization_15_assignmovingavg_readvariableop_resource.batch_normalization_15/AssignMovingAvg/mul:z:06^batch_normalization_15/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_15/AssignMovingAvg�
.batch_normalization_15/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_15/AssignMovingAvg_1/decay�
-batch_normalization_15/AssignMovingAvg_1/CastCast7batch_normalization_15/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_15/AssignMovingAvg_1/Cast�
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_15_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_15/AssignMovingAvg_1/subSub?batch_normalization_15/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_15/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_15/AssignMovingAvg_1/sub�
,batch_normalization_15/AssignMovingAvg_1/mulMul0batch_normalization_15/AssignMovingAvg_1/sub:z:01batch_normalization_15/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_15/AssignMovingAvg_1/mul�
(batch_normalization_15/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_15_assignmovingavg_1_readvariableop_resource0batch_normalization_15/AssignMovingAvg_1/mul:z:08^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_15/AssignMovingAvg_1�
*batch_normalization_15/Cast/ReadVariableOpReadVariableOp3batch_normalization_15_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_15/Cast/ReadVariableOp�
,batch_normalization_15/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_15_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_15/Cast_1/ReadVariableOp�
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_15/batchnorm/add/y�
$batch_normalization_15/batchnorm/addAddV21batch_normalization_15/moments/Squeeze_1:output:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/add�
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/Rsqrt�
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:04batch_normalization_15/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/mul�
&batch_normalization_15/batchnorm/mul_1Muldense_12/MatMul:product:0(batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_15/batchnorm/mul_1�
&batch_normalization_15/batchnorm/mul_2Mul/batch_normalization_15/moments/Squeeze:output:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/mul_2�
$batch_normalization_15/batchnorm/subSub2batch_normalization_15/Cast/ReadVariableOp:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/sub�
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_15/batchnorm/add_1v
Relu_2Relu*batch_normalization_15/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulRelu_2:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_13/MatMul�
5batch_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_16/moments/mean/reduction_indices�
#batch_normalization_16/moments/meanMeandense_13/MatMul:product:0>batch_normalization_16/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_16/moments/mean�
+batch_normalization_16/moments/StopGradientStopGradient,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_16/moments/StopGradient�
0batch_normalization_16/moments/SquaredDifferenceSquaredDifferencedense_13/MatMul:product:04batch_normalization_16/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_16/moments/SquaredDifference�
9batch_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_16/moments/variance/reduction_indices�
'batch_normalization_16/moments/varianceMean4batch_normalization_16/moments/SquaredDifference:z:0Bbatch_normalization_16/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_16/moments/variance�
&batch_normalization_16/moments/SqueezeSqueeze,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_16/moments/Squeeze�
(batch_normalization_16/moments/Squeeze_1Squeeze0batch_normalization_16/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_16/moments/Squeeze_1�
,batch_normalization_16/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_16/AssignMovingAvg/decay�
+batch_normalization_16/AssignMovingAvg/CastCast5batch_normalization_16/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_16/AssignMovingAvg/Cast�
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_16/AssignMovingAvg/ReadVariableOp�
*batch_normalization_16/AssignMovingAvg/subSub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_16/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_16/AssignMovingAvg/sub�
*batch_normalization_16/AssignMovingAvg/mulMul.batch_normalization_16/AssignMovingAvg/sub:z:0/batch_normalization_16/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_16/AssignMovingAvg/mul�
&batch_normalization_16/AssignMovingAvgAssignSubVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_16/AssignMovingAvg�
.batch_normalization_16/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_16/AssignMovingAvg_1/decay�
-batch_normalization_16/AssignMovingAvg_1/CastCast7batch_normalization_16/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_16/AssignMovingAvg_1/Cast�
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_16/AssignMovingAvg_1/subSub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_16/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_16/AssignMovingAvg_1/sub�
,batch_normalization_16/AssignMovingAvg_1/mulMul0batch_normalization_16/AssignMovingAvg_1/sub:z:01batch_normalization_16/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_16/AssignMovingAvg_1/mul�
(batch_normalization_16/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource0batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_16/AssignMovingAvg_1�
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_16/Cast/ReadVariableOp�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_16/Cast_1/ReadVariableOp�
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_16/batchnorm/add/y�
$batch_normalization_16/batchnorm/addAddV21batch_normalization_16/moments/Squeeze_1:output:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/add�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/Rsqrt�
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:04batch_normalization_16/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/mul�
&batch_normalization_16/batchnorm/mul_1Muldense_13/MatMul:product:0(batch_normalization_16/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_16/batchnorm/mul_1�
&batch_normalization_16/batchnorm/mul_2Mul/batch_normalization_16/moments/Squeeze:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/mul_2�
$batch_normalization_16/batchnorm/subSub2batch_normalization_16/Cast/ReadVariableOp:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/sub�
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_16/batchnorm/add_1v
Relu_3Relu*batch_normalization_16/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMulRelu_3:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_14/BiasAddt
IdentityIdentitydense_14/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp'^batch_normalization_12/AssignMovingAvg6^batch_normalization_12/AssignMovingAvg/ReadVariableOp)^batch_normalization_12/AssignMovingAvg_18^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_12/Cast/ReadVariableOp-^batch_normalization_12/Cast_1/ReadVariableOp'^batch_normalization_13/AssignMovingAvg6^batch_normalization_13/AssignMovingAvg/ReadVariableOp)^batch_normalization_13/AssignMovingAvg_18^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_13/Cast/ReadVariableOp-^batch_normalization_13/Cast_1/ReadVariableOp'^batch_normalization_14/AssignMovingAvg6^batch_normalization_14/AssignMovingAvg/ReadVariableOp)^batch_normalization_14/AssignMovingAvg_18^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_14/Cast/ReadVariableOp-^batch_normalization_14/Cast_1/ReadVariableOp'^batch_normalization_15/AssignMovingAvg6^batch_normalization_15/AssignMovingAvg/ReadVariableOp)^batch_normalization_15/AssignMovingAvg_18^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_15/Cast/ReadVariableOp-^batch_normalization_15/Cast_1/ReadVariableOp'^batch_normalization_16/AssignMovingAvg6^batch_normalization_16/AssignMovingAvg/ReadVariableOp)^batch_normalization_16/AssignMovingAvg_18^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_12/MatMul/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_12/AssignMovingAvg&batch_normalization_12/AssignMovingAvg2n
5batch_normalization_12/AssignMovingAvg/ReadVariableOp5batch_normalization_12/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_12/AssignMovingAvg_1(batch_normalization_12/AssignMovingAvg_12r
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_12/Cast/ReadVariableOp*batch_normalization_12/Cast/ReadVariableOp2\
,batch_normalization_12/Cast_1/ReadVariableOp,batch_normalization_12/Cast_1/ReadVariableOp2P
&batch_normalization_13/AssignMovingAvg&batch_normalization_13/AssignMovingAvg2n
5batch_normalization_13/AssignMovingAvg/ReadVariableOp5batch_normalization_13/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_13/AssignMovingAvg_1(batch_normalization_13/AssignMovingAvg_12r
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_13/Cast/ReadVariableOp*batch_normalization_13/Cast/ReadVariableOp2\
,batch_normalization_13/Cast_1/ReadVariableOp,batch_normalization_13/Cast_1/ReadVariableOp2P
&batch_normalization_14/AssignMovingAvg&batch_normalization_14/AssignMovingAvg2n
5batch_normalization_14/AssignMovingAvg/ReadVariableOp5batch_normalization_14/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_14/AssignMovingAvg_1(batch_normalization_14/AssignMovingAvg_12r
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_14/Cast/ReadVariableOp*batch_normalization_14/Cast/ReadVariableOp2\
,batch_normalization_14/Cast_1/ReadVariableOp,batch_normalization_14/Cast_1/ReadVariableOp2P
&batch_normalization_15/AssignMovingAvg&batch_normalization_15/AssignMovingAvg2n
5batch_normalization_15/AssignMovingAvg/ReadVariableOp5batch_normalization_15/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_15/AssignMovingAvg_1(batch_normalization_15/AssignMovingAvg_12r
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_15/Cast/ReadVariableOp*batch_normalization_15/Cast/ReadVariableOp2\
,batch_normalization_15/Cast_1/ReadVariableOp,batch_normalization_15/Cast_1/ReadVariableOp2P
&batch_normalization_16/AssignMovingAvg&batch_normalization_16/AssignMovingAvg2n
5batch_normalization_16/AssignMovingAvg/ReadVariableOp5batch_normalization_16/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_16/AssignMovingAvg_1(batch_normalization_16/AssignMovingAvg_12r
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������

!
_user_specified_name	input_1
�

�
D__inference_dense_14_layer_call_and_return_conditional_losses_315942

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
�+
�
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_315703

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
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_314334
x+
batch_normalization_12_314267:
+
batch_normalization_12_314269:
+
batch_normalization_12_314271:
+
batch_normalization_12_314273:
!
dense_10_314276:
+
batch_normalization_13_314279:+
batch_normalization_13_314281:+
batch_normalization_13_314283:+
batch_normalization_13_314285:!
dense_11_314289:+
batch_normalization_14_314292:+
batch_normalization_14_314294:+
batch_normalization_14_314296:+
batch_normalization_14_314298:!
dense_12_314302:+
batch_normalization_15_314305:+
batch_normalization_15_314307:+
batch_normalization_15_314309:+
batch_normalization_15_314311:!
dense_13_314315:+
batch_normalization_16_314318:+
batch_normalization_16_314320:+
batch_normalization_16_314322:+
batch_normalization_16_314324:!
dense_14_314328:

dense_14_314330:

identity��.batch_normalization_12/StatefulPartitionedCall�.batch_normalization_13/StatefulPartitionedCall�.batch_normalization_14/StatefulPartitionedCall�.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_12_314267batch_normalization_12_314269batch_normalization_12_314271batch_normalization_12_314273*
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
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_31324720
.batch_normalization_12/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0dense_10_314276*
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
D__inference_dense_10_layer_call_and_return_conditional_losses_3140142"
 dense_10/StatefulPartitionedCall�
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_13_314279batch_normalization_13_314281batch_normalization_13_314283batch_normalization_13_314285*
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
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_31341320
.batch_normalization_13/StatefulPartitionedCall
ReluRelu7batch_normalization_13/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu�
 dense_11/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_11_314289*
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
D__inference_dense_11_layer_call_and_return_conditional_losses_3140352"
 dense_11/StatefulPartitionedCall�
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0batch_normalization_14_314292batch_normalization_14_314294batch_normalization_14_314296batch_normalization_14_314298*
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
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_31357920
.batch_normalization_14/StatefulPartitionedCall�
Relu_1Relu7batch_normalization_14/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_1�
 dense_12/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_12_314302*
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
D__inference_dense_12_layer_call_and_return_conditional_losses_3140562"
 dense_12/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0batch_normalization_15_314305batch_normalization_15_314307batch_normalization_15_314309batch_normalization_15_314311*
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
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_31374520
.batch_normalization_15/StatefulPartitionedCall�
Relu_2Relu7batch_normalization_15/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_2�
 dense_13/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_13_314315*
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
D__inference_dense_13_layer_call_and_return_conditional_losses_3140772"
 dense_13/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0batch_normalization_16_314318batch_normalization_16_314320batch_normalization_16_314322batch_normalization_16_314324*
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
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_31391120
.batch_normalization_16/StatefulPartitionedCall�
Relu_3Relu7batch_normalization_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_3�
 dense_14/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_14_314328dense_14_314330*
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
D__inference_dense_14_layer_call_and_return_conditional_losses_3141012"
 dense_14/StatefulPartitionedCall�
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:J F
'
_output_shapes
:���������


_user_specified_namex
�+
�
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_315785

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
D__inference_dense_11_layer_call_and_return_conditional_losses_315895

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
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_313351

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
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_313683

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
7__inference_batch_normalization_15_layer_call_fn_315716

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
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3136832
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
�
�
7__inference_feed_forward_sub_net_2_layer_call_fn_314873
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
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_3143342
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
��
�
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_315165
xL
>batch_normalization_12_assignmovingavg_readvariableop_resource:
N
@batch_normalization_12_assignmovingavg_1_readvariableop_resource:
A
3batch_normalization_12_cast_readvariableop_resource:
C
5batch_normalization_12_cast_1_readvariableop_resource:
9
'dense_10_matmul_readvariableop_resource:
L
>batch_normalization_13_assignmovingavg_readvariableop_resource:N
@batch_normalization_13_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_13_cast_readvariableop_resource:C
5batch_normalization_13_cast_1_readvariableop_resource:9
'dense_11_matmul_readvariableop_resource:L
>batch_normalization_14_assignmovingavg_readvariableop_resource:N
@batch_normalization_14_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_14_cast_readvariableop_resource:C
5batch_normalization_14_cast_1_readvariableop_resource:9
'dense_12_matmul_readvariableop_resource:L
>batch_normalization_15_assignmovingavg_readvariableop_resource:N
@batch_normalization_15_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_15_cast_readvariableop_resource:C
5batch_normalization_15_cast_1_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:L
>batch_normalization_16_assignmovingavg_readvariableop_resource:N
@batch_normalization_16_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_16_cast_readvariableop_resource:C
5batch_normalization_16_cast_1_readvariableop_resource:9
'dense_14_matmul_readvariableop_resource:
6
(dense_14_biasadd_readvariableop_resource:

identity��&batch_normalization_12/AssignMovingAvg�5batch_normalization_12/AssignMovingAvg/ReadVariableOp�(batch_normalization_12/AssignMovingAvg_1�7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_12/Cast/ReadVariableOp�,batch_normalization_12/Cast_1/ReadVariableOp�&batch_normalization_13/AssignMovingAvg�5batch_normalization_13/AssignMovingAvg/ReadVariableOp�(batch_normalization_13/AssignMovingAvg_1�7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_13/Cast/ReadVariableOp�,batch_normalization_13/Cast_1/ReadVariableOp�&batch_normalization_14/AssignMovingAvg�5batch_normalization_14/AssignMovingAvg/ReadVariableOp�(batch_normalization_14/AssignMovingAvg_1�7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_14/Cast/ReadVariableOp�,batch_normalization_14/Cast_1/ReadVariableOp�&batch_normalization_15/AssignMovingAvg�5batch_normalization_15/AssignMovingAvg/ReadVariableOp�(batch_normalization_15/AssignMovingAvg_1�7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_15/Cast/ReadVariableOp�,batch_normalization_15/Cast_1/ReadVariableOp�&batch_normalization_16/AssignMovingAvg�5batch_normalization_16/AssignMovingAvg/ReadVariableOp�(batch_normalization_16/AssignMovingAvg_1�7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_16/Cast/ReadVariableOp�,batch_normalization_16/Cast_1/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
5batch_normalization_12/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_12/moments/mean/reduction_indices�
#batch_normalization_12/moments/meanMeanx>batch_normalization_12/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_12/moments/mean�
+batch_normalization_12/moments/StopGradientStopGradient,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_12/moments/StopGradient�
0batch_normalization_12/moments/SquaredDifferenceSquaredDifferencex4batch_normalization_12/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������
22
0batch_normalization_12/moments/SquaredDifference�
9batch_normalization_12/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_12/moments/variance/reduction_indices�
'batch_normalization_12/moments/varianceMean4batch_normalization_12/moments/SquaredDifference:z:0Bbatch_normalization_12/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_12/moments/variance�
&batch_normalization_12/moments/SqueezeSqueeze,batch_normalization_12/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_12/moments/Squeeze�
(batch_normalization_12/moments/Squeeze_1Squeeze0batch_normalization_12/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_12/moments/Squeeze_1�
,batch_normalization_12/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_12/AssignMovingAvg/decay�
+batch_normalization_12/AssignMovingAvg/CastCast5batch_normalization_12/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_12/AssignMovingAvg/Cast�
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_12_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_12/AssignMovingAvg/ReadVariableOp�
*batch_normalization_12/AssignMovingAvg/subSub=batch_normalization_12/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_12/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_12/AssignMovingAvg/sub�
*batch_normalization_12/AssignMovingAvg/mulMul.batch_normalization_12/AssignMovingAvg/sub:z:0/batch_normalization_12/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_12/AssignMovingAvg/mul�
&batch_normalization_12/AssignMovingAvgAssignSubVariableOp>batch_normalization_12_assignmovingavg_readvariableop_resource.batch_normalization_12/AssignMovingAvg/mul:z:06^batch_normalization_12/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_12/AssignMovingAvg�
.batch_normalization_12/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_12/AssignMovingAvg_1/decay�
-batch_normalization_12/AssignMovingAvg_1/CastCast7batch_normalization_12/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_12/AssignMovingAvg_1/Cast�
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_12_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_12/AssignMovingAvg_1/subSub?batch_normalization_12/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_12/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_12/AssignMovingAvg_1/sub�
,batch_normalization_12/AssignMovingAvg_1/mulMul0batch_normalization_12/AssignMovingAvg_1/sub:z:01batch_normalization_12/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_12/AssignMovingAvg_1/mul�
(batch_normalization_12/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_12_assignmovingavg_1_readvariableop_resource0batch_normalization_12/AssignMovingAvg_1/mul:z:08^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_12/AssignMovingAvg_1�
*batch_normalization_12/Cast/ReadVariableOpReadVariableOp3batch_normalization_12_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_12/Cast/ReadVariableOp�
,batch_normalization_12/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_12_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_12/Cast_1/ReadVariableOp�
&batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_12/batchnorm/add/y�
$batch_normalization_12/batchnorm/addAddV21batch_normalization_12/moments/Squeeze_1:output:0/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_12/batchnorm/add�
&batch_normalization_12/batchnorm/RsqrtRsqrt(batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_12/batchnorm/Rsqrt�
$batch_normalization_12/batchnorm/mulMul*batch_normalization_12/batchnorm/Rsqrt:y:04batch_normalization_12/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_12/batchnorm/mul�
&batch_normalization_12/batchnorm/mul_1Mulx(batch_normalization_12/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_12/batchnorm/mul_1�
&batch_normalization_12/batchnorm/mul_2Mul/batch_normalization_12/moments/Squeeze:output:0(batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_12/batchnorm/mul_2�
$batch_normalization_12/batchnorm/subSub2batch_normalization_12/Cast/ReadVariableOp:value:0*batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_12/batchnorm/sub�
&batch_normalization_12/batchnorm/add_1AddV2*batch_normalization_12/batchnorm/mul_1:z:0(batch_normalization_12/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������
2(
&batch_normalization_12/batchnorm/add_1�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMul*batch_normalization_12/batchnorm/add_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_10/MatMul�
5batch_normalization_13/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_13/moments/mean/reduction_indices�
#batch_normalization_13/moments/meanMeandense_10/MatMul:product:0>batch_normalization_13/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_13/moments/mean�
+batch_normalization_13/moments/StopGradientStopGradient,batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_13/moments/StopGradient�
0batch_normalization_13/moments/SquaredDifferenceSquaredDifferencedense_10/MatMul:product:04batch_normalization_13/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_13/moments/SquaredDifference�
9batch_normalization_13/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_13/moments/variance/reduction_indices�
'batch_normalization_13/moments/varianceMean4batch_normalization_13/moments/SquaredDifference:z:0Bbatch_normalization_13/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_13/moments/variance�
&batch_normalization_13/moments/SqueezeSqueeze,batch_normalization_13/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_13/moments/Squeeze�
(batch_normalization_13/moments/Squeeze_1Squeeze0batch_normalization_13/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_13/moments/Squeeze_1�
,batch_normalization_13/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_13/AssignMovingAvg/decay�
+batch_normalization_13/AssignMovingAvg/CastCast5batch_normalization_13/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_13/AssignMovingAvg/Cast�
5batch_normalization_13/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_13_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_13/AssignMovingAvg/ReadVariableOp�
*batch_normalization_13/AssignMovingAvg/subSub=batch_normalization_13/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_13/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_13/AssignMovingAvg/sub�
*batch_normalization_13/AssignMovingAvg/mulMul.batch_normalization_13/AssignMovingAvg/sub:z:0/batch_normalization_13/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_13/AssignMovingAvg/mul�
&batch_normalization_13/AssignMovingAvgAssignSubVariableOp>batch_normalization_13_assignmovingavg_readvariableop_resource.batch_normalization_13/AssignMovingAvg/mul:z:06^batch_normalization_13/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_13/AssignMovingAvg�
.batch_normalization_13/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_13/AssignMovingAvg_1/decay�
-batch_normalization_13/AssignMovingAvg_1/CastCast7batch_normalization_13/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_13/AssignMovingAvg_1/Cast�
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_13_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_13/AssignMovingAvg_1/subSub?batch_normalization_13/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_13/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_13/AssignMovingAvg_1/sub�
,batch_normalization_13/AssignMovingAvg_1/mulMul0batch_normalization_13/AssignMovingAvg_1/sub:z:01batch_normalization_13/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_13/AssignMovingAvg_1/mul�
(batch_normalization_13/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_13_assignmovingavg_1_readvariableop_resource0batch_normalization_13/AssignMovingAvg_1/mul:z:08^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_13/AssignMovingAvg_1�
*batch_normalization_13/Cast/ReadVariableOpReadVariableOp3batch_normalization_13_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_13/Cast/ReadVariableOp�
,batch_normalization_13/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_13_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_13/Cast_1/ReadVariableOp�
&batch_normalization_13/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_13/batchnorm/add/y�
$batch_normalization_13/batchnorm/addAddV21batch_normalization_13/moments/Squeeze_1:output:0/batch_normalization_13/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/add�
&batch_normalization_13/batchnorm/RsqrtRsqrt(batch_normalization_13/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/Rsqrt�
$batch_normalization_13/batchnorm/mulMul*batch_normalization_13/batchnorm/Rsqrt:y:04batch_normalization_13/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/mul�
&batch_normalization_13/batchnorm/mul_1Muldense_10/MatMul:product:0(batch_normalization_13/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_13/batchnorm/mul_1�
&batch_normalization_13/batchnorm/mul_2Mul/batch_normalization_13/moments/Squeeze:output:0(batch_normalization_13/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_13/batchnorm/mul_2�
$batch_normalization_13/batchnorm/subSub2batch_normalization_13/Cast/ReadVariableOp:value:0*batch_normalization_13/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_13/batchnorm/sub�
&batch_normalization_13/batchnorm/add_1AddV2*batch_normalization_13/batchnorm/mul_1:z:0(batch_normalization_13/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_13/batchnorm/add_1r
ReluRelu*batch_normalization_13/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMulRelu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
5batch_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_14/moments/mean/reduction_indices�
#batch_normalization_14/moments/meanMeandense_11/MatMul:product:0>batch_normalization_14/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_14/moments/mean�
+batch_normalization_14/moments/StopGradientStopGradient,batch_normalization_14/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_14/moments/StopGradient�
0batch_normalization_14/moments/SquaredDifferenceSquaredDifferencedense_11/MatMul:product:04batch_normalization_14/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_14/moments/SquaredDifference�
9batch_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_14/moments/variance/reduction_indices�
'batch_normalization_14/moments/varianceMean4batch_normalization_14/moments/SquaredDifference:z:0Bbatch_normalization_14/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_14/moments/variance�
&batch_normalization_14/moments/SqueezeSqueeze,batch_normalization_14/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_14/moments/Squeeze�
(batch_normalization_14/moments/Squeeze_1Squeeze0batch_normalization_14/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_14/moments/Squeeze_1�
,batch_normalization_14/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_14/AssignMovingAvg/decay�
+batch_normalization_14/AssignMovingAvg/CastCast5batch_normalization_14/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_14/AssignMovingAvg/Cast�
5batch_normalization_14/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_14_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_14/AssignMovingAvg/ReadVariableOp�
*batch_normalization_14/AssignMovingAvg/subSub=batch_normalization_14/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_14/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_14/AssignMovingAvg/sub�
*batch_normalization_14/AssignMovingAvg/mulMul.batch_normalization_14/AssignMovingAvg/sub:z:0/batch_normalization_14/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_14/AssignMovingAvg/mul�
&batch_normalization_14/AssignMovingAvgAssignSubVariableOp>batch_normalization_14_assignmovingavg_readvariableop_resource.batch_normalization_14/AssignMovingAvg/mul:z:06^batch_normalization_14/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_14/AssignMovingAvg�
.batch_normalization_14/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_14/AssignMovingAvg_1/decay�
-batch_normalization_14/AssignMovingAvg_1/CastCast7batch_normalization_14/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_14/AssignMovingAvg_1/Cast�
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_14_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_14/AssignMovingAvg_1/subSub?batch_normalization_14/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_14/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_14/AssignMovingAvg_1/sub�
,batch_normalization_14/AssignMovingAvg_1/mulMul0batch_normalization_14/AssignMovingAvg_1/sub:z:01batch_normalization_14/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_14/AssignMovingAvg_1/mul�
(batch_normalization_14/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_14_assignmovingavg_1_readvariableop_resource0batch_normalization_14/AssignMovingAvg_1/mul:z:08^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_14/AssignMovingAvg_1�
*batch_normalization_14/Cast/ReadVariableOpReadVariableOp3batch_normalization_14_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_14/Cast/ReadVariableOp�
,batch_normalization_14/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_14_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_14/Cast_1/ReadVariableOp�
&batch_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_14/batchnorm/add/y�
$batch_normalization_14/batchnorm/addAddV21batch_normalization_14/moments/Squeeze_1:output:0/batch_normalization_14/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/add�
&batch_normalization_14/batchnorm/RsqrtRsqrt(batch_normalization_14/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/Rsqrt�
$batch_normalization_14/batchnorm/mulMul*batch_normalization_14/batchnorm/Rsqrt:y:04batch_normalization_14/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/mul�
&batch_normalization_14/batchnorm/mul_1Muldense_11/MatMul:product:0(batch_normalization_14/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_14/batchnorm/mul_1�
&batch_normalization_14/batchnorm/mul_2Mul/batch_normalization_14/moments/Squeeze:output:0(batch_normalization_14/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_14/batchnorm/mul_2�
$batch_normalization_14/batchnorm/subSub2batch_normalization_14/Cast/ReadVariableOp:value:0*batch_normalization_14/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_14/batchnorm/sub�
&batch_normalization_14/batchnorm/add_1AddV2*batch_normalization_14/batchnorm/mul_1:z:0(batch_normalization_14/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_14/batchnorm/add_1v
Relu_1Relu*batch_normalization_14/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_1�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulRelu_1:activations:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_12/MatMul�
5batch_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_15/moments/mean/reduction_indices�
#batch_normalization_15/moments/meanMeandense_12/MatMul:product:0>batch_normalization_15/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_15/moments/mean�
+batch_normalization_15/moments/StopGradientStopGradient,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_15/moments/StopGradient�
0batch_normalization_15/moments/SquaredDifferenceSquaredDifferencedense_12/MatMul:product:04batch_normalization_15/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_15/moments/SquaredDifference�
9batch_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_15/moments/variance/reduction_indices�
'batch_normalization_15/moments/varianceMean4batch_normalization_15/moments/SquaredDifference:z:0Bbatch_normalization_15/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_15/moments/variance�
&batch_normalization_15/moments/SqueezeSqueeze,batch_normalization_15/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_15/moments/Squeeze�
(batch_normalization_15/moments/Squeeze_1Squeeze0batch_normalization_15/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_15/moments/Squeeze_1�
,batch_normalization_15/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_15/AssignMovingAvg/decay�
+batch_normalization_15/AssignMovingAvg/CastCast5batch_normalization_15/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_15/AssignMovingAvg/Cast�
5batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_15_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_15/AssignMovingAvg/ReadVariableOp�
*batch_normalization_15/AssignMovingAvg/subSub=batch_normalization_15/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_15/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_15/AssignMovingAvg/sub�
*batch_normalization_15/AssignMovingAvg/mulMul.batch_normalization_15/AssignMovingAvg/sub:z:0/batch_normalization_15/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_15/AssignMovingAvg/mul�
&batch_normalization_15/AssignMovingAvgAssignSubVariableOp>batch_normalization_15_assignmovingavg_readvariableop_resource.batch_normalization_15/AssignMovingAvg/mul:z:06^batch_normalization_15/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_15/AssignMovingAvg�
.batch_normalization_15/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_15/AssignMovingAvg_1/decay�
-batch_normalization_15/AssignMovingAvg_1/CastCast7batch_normalization_15/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_15/AssignMovingAvg_1/Cast�
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_15_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_15/AssignMovingAvg_1/subSub?batch_normalization_15/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_15/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_15/AssignMovingAvg_1/sub�
,batch_normalization_15/AssignMovingAvg_1/mulMul0batch_normalization_15/AssignMovingAvg_1/sub:z:01batch_normalization_15/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_15/AssignMovingAvg_1/mul�
(batch_normalization_15/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_15_assignmovingavg_1_readvariableop_resource0batch_normalization_15/AssignMovingAvg_1/mul:z:08^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_15/AssignMovingAvg_1�
*batch_normalization_15/Cast/ReadVariableOpReadVariableOp3batch_normalization_15_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_15/Cast/ReadVariableOp�
,batch_normalization_15/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_15_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_15/Cast_1/ReadVariableOp�
&batch_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_15/batchnorm/add/y�
$batch_normalization_15/batchnorm/addAddV21batch_normalization_15/moments/Squeeze_1:output:0/batch_normalization_15/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/add�
&batch_normalization_15/batchnorm/RsqrtRsqrt(batch_normalization_15/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/Rsqrt�
$batch_normalization_15/batchnorm/mulMul*batch_normalization_15/batchnorm/Rsqrt:y:04batch_normalization_15/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/mul�
&batch_normalization_15/batchnorm/mul_1Muldense_12/MatMul:product:0(batch_normalization_15/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_15/batchnorm/mul_1�
&batch_normalization_15/batchnorm/mul_2Mul/batch_normalization_15/moments/Squeeze:output:0(batch_normalization_15/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_15/batchnorm/mul_2�
$batch_normalization_15/batchnorm/subSub2batch_normalization_15/Cast/ReadVariableOp:value:0*batch_normalization_15/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_15/batchnorm/sub�
&batch_normalization_15/batchnorm/add_1AddV2*batch_normalization_15/batchnorm/mul_1:z:0(batch_normalization_15/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_15/batchnorm/add_1v
Relu_2Relu*batch_normalization_15/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_2�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulRelu_2:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_13/MatMul�
5batch_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_16/moments/mean/reduction_indices�
#batch_normalization_16/moments/meanMeandense_13/MatMul:product:0>batch_normalization_16/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_16/moments/mean�
+batch_normalization_16/moments/StopGradientStopGradient,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_16/moments/StopGradient�
0batch_normalization_16/moments/SquaredDifferenceSquaredDifferencedense_13/MatMul:product:04batch_normalization_16/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������22
0batch_normalization_16/moments/SquaredDifference�
9batch_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_16/moments/variance/reduction_indices�
'batch_normalization_16/moments/varianceMean4batch_normalization_16/moments/SquaredDifference:z:0Bbatch_normalization_16/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_16/moments/variance�
&batch_normalization_16/moments/SqueezeSqueeze,batch_normalization_16/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_16/moments/Squeeze�
(batch_normalization_16/moments/Squeeze_1Squeeze0batch_normalization_16/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_16/moments/Squeeze_1�
,batch_normalization_16/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_16/AssignMovingAvg/decay�
+batch_normalization_16/AssignMovingAvg/CastCast5batch_normalization_16/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_16/AssignMovingAvg/Cast�
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_16/AssignMovingAvg/ReadVariableOp�
*batch_normalization_16/AssignMovingAvg/subSub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_16/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_16/AssignMovingAvg/sub�
*batch_normalization_16/AssignMovingAvg/mulMul.batch_normalization_16/AssignMovingAvg/sub:z:0/batch_normalization_16/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_16/AssignMovingAvg/mul�
&batch_normalization_16/AssignMovingAvgAssignSubVariableOp>batch_normalization_16_assignmovingavg_readvariableop_resource.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_16/AssignMovingAvg�
.batch_normalization_16/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_16/AssignMovingAvg_1/decay�
-batch_normalization_16/AssignMovingAvg_1/CastCast7batch_normalization_16/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_16/AssignMovingAvg_1/Cast�
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_16/AssignMovingAvg_1/subSub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_16/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_16/AssignMovingAvg_1/sub�
,batch_normalization_16/AssignMovingAvg_1/mulMul0batch_normalization_16/AssignMovingAvg_1/sub:z:01batch_normalization_16/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_16/AssignMovingAvg_1/mul�
(batch_normalization_16/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_16_assignmovingavg_1_readvariableop_resource0batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_16/AssignMovingAvg_1�
*batch_normalization_16/Cast/ReadVariableOpReadVariableOp3batch_normalization_16_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_16/Cast/ReadVariableOp�
,batch_normalization_16/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_16_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_16/Cast_1/ReadVariableOp�
&batch_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_16/batchnorm/add/y�
$batch_normalization_16/batchnorm/addAddV21batch_normalization_16/moments/Squeeze_1:output:0/batch_normalization_16/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/add�
&batch_normalization_16/batchnorm/RsqrtRsqrt(batch_normalization_16/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/Rsqrt�
$batch_normalization_16/batchnorm/mulMul*batch_normalization_16/batchnorm/Rsqrt:y:04batch_normalization_16/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/mul�
&batch_normalization_16/batchnorm/mul_1Muldense_13/MatMul:product:0(batch_normalization_16/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_16/batchnorm/mul_1�
&batch_normalization_16/batchnorm/mul_2Mul/batch_normalization_16/moments/Squeeze:output:0(batch_normalization_16/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_16/batchnorm/mul_2�
$batch_normalization_16/batchnorm/subSub2batch_normalization_16/Cast/ReadVariableOp:value:0*batch_normalization_16/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_16/batchnorm/sub�
&batch_normalization_16/batchnorm/add_1AddV2*batch_normalization_16/batchnorm/mul_1:z:0(batch_normalization_16/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������2(
&batch_normalization_16/batchnorm/add_1v
Relu_3Relu*batch_normalization_16/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������2
Relu_3�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMulRelu_3:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_14/BiasAddt
IdentityIdentitydense_14/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp'^batch_normalization_12/AssignMovingAvg6^batch_normalization_12/AssignMovingAvg/ReadVariableOp)^batch_normalization_12/AssignMovingAvg_18^batch_normalization_12/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_12/Cast/ReadVariableOp-^batch_normalization_12/Cast_1/ReadVariableOp'^batch_normalization_13/AssignMovingAvg6^batch_normalization_13/AssignMovingAvg/ReadVariableOp)^batch_normalization_13/AssignMovingAvg_18^batch_normalization_13/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_13/Cast/ReadVariableOp-^batch_normalization_13/Cast_1/ReadVariableOp'^batch_normalization_14/AssignMovingAvg6^batch_normalization_14/AssignMovingAvg/ReadVariableOp)^batch_normalization_14/AssignMovingAvg_18^batch_normalization_14/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_14/Cast/ReadVariableOp-^batch_normalization_14/Cast_1/ReadVariableOp'^batch_normalization_15/AssignMovingAvg6^batch_normalization_15/AssignMovingAvg/ReadVariableOp)^batch_normalization_15/AssignMovingAvg_18^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_15/Cast/ReadVariableOp-^batch_normalization_15/Cast_1/ReadVariableOp'^batch_normalization_16/AssignMovingAvg6^batch_normalization_16/AssignMovingAvg/ReadVariableOp)^batch_normalization_16/AssignMovingAvg_18^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_16/Cast/ReadVariableOp-^batch_normalization_16/Cast_1/ReadVariableOp^dense_10/MatMul/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_12/MatMul/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_12/AssignMovingAvg&batch_normalization_12/AssignMovingAvg2n
5batch_normalization_12/AssignMovingAvg/ReadVariableOp5batch_normalization_12/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_12/AssignMovingAvg_1(batch_normalization_12/AssignMovingAvg_12r
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_12/Cast/ReadVariableOp*batch_normalization_12/Cast/ReadVariableOp2\
,batch_normalization_12/Cast_1/ReadVariableOp,batch_normalization_12/Cast_1/ReadVariableOp2P
&batch_normalization_13/AssignMovingAvg&batch_normalization_13/AssignMovingAvg2n
5batch_normalization_13/AssignMovingAvg/ReadVariableOp5batch_normalization_13/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_13/AssignMovingAvg_1(batch_normalization_13/AssignMovingAvg_12r
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_13/Cast/ReadVariableOp*batch_normalization_13/Cast/ReadVariableOp2\
,batch_normalization_13/Cast_1/ReadVariableOp,batch_normalization_13/Cast_1/ReadVariableOp2P
&batch_normalization_14/AssignMovingAvg&batch_normalization_14/AssignMovingAvg2n
5batch_normalization_14/AssignMovingAvg/ReadVariableOp5batch_normalization_14/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_14/AssignMovingAvg_1(batch_normalization_14/AssignMovingAvg_12r
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_14/Cast/ReadVariableOp*batch_normalization_14/Cast/ReadVariableOp2\
,batch_normalization_14/Cast_1/ReadVariableOp,batch_normalization_14/Cast_1/ReadVariableOp2P
&batch_normalization_15/AssignMovingAvg&batch_normalization_15/AssignMovingAvg2n
5batch_normalization_15/AssignMovingAvg/ReadVariableOp5batch_normalization_15/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_15/AssignMovingAvg_1(batch_normalization_15/AssignMovingAvg_12r
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_15/Cast/ReadVariableOp*batch_normalization_15/Cast/ReadVariableOp2\
,batch_normalization_15/Cast_1/ReadVariableOp,batch_normalization_15/Cast_1/ReadVariableOp2P
&batch_normalization_16/AssignMovingAvg&batch_normalization_16/AssignMovingAvg2n
5batch_normalization_16/AssignMovingAvg/ReadVariableOp5batch_normalization_16/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_16/AssignMovingAvg_1(batch_normalization_16/AssignMovingAvg_12r
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_16/Cast/ReadVariableOp*batch_normalization_16/Cast/ReadVariableOp2\
,batch_normalization_16/Cast_1/ReadVariableOp,batch_normalization_16/Cast_1/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������


_user_specified_namex
�+
�
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_313579

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
)__inference_dense_12_layer_call_fn_315902

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
D__inference_dense_12_layer_call_and_return_conditional_losses_3140562
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
7__inference_batch_normalization_15_layer_call_fn_315729

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
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_3137452
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
�
)__inference_dense_14_layer_call_fn_315932

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
D__inference_dense_14_layer_call_and_return_conditional_losses_3141012
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
�
�
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_315749

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
�D
�
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_314108
x+
batch_normalization_12_313998:
+
batch_normalization_12_314000:
+
batch_normalization_12_314002:
+
batch_normalization_12_314004:
!
dense_10_314015:
+
batch_normalization_13_314018:+
batch_normalization_13_314020:+
batch_normalization_13_314022:+
batch_normalization_13_314024:!
dense_11_314036:+
batch_normalization_14_314039:+
batch_normalization_14_314041:+
batch_normalization_14_314043:+
batch_normalization_14_314045:!
dense_12_314057:+
batch_normalization_15_314060:+
batch_normalization_15_314062:+
batch_normalization_15_314064:+
batch_normalization_15_314066:!
dense_13_314078:+
batch_normalization_16_314081:+
batch_normalization_16_314083:+
batch_normalization_16_314085:+
batch_normalization_16_314087:!
dense_14_314102:

dense_14_314104:

identity��.batch_normalization_12/StatefulPartitionedCall�.batch_normalization_13/StatefulPartitionedCall�.batch_normalization_14/StatefulPartitionedCall�.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_12_313998batch_normalization_12_314000batch_normalization_12_314002batch_normalization_12_314004*
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
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_31318520
.batch_normalization_12/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0dense_10_314015*
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
D__inference_dense_10_layer_call_and_return_conditional_losses_3140142"
 dense_10/StatefulPartitionedCall�
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0batch_normalization_13_314018batch_normalization_13_314020batch_normalization_13_314022batch_normalization_13_314024*
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
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_31335120
.batch_normalization_13/StatefulPartitionedCall
ReluRelu7batch_normalization_13/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu�
 dense_11/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_11_314036*
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
D__inference_dense_11_layer_call_and_return_conditional_losses_3140352"
 dense_11/StatefulPartitionedCall�
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0batch_normalization_14_314039batch_normalization_14_314041batch_normalization_14_314043batch_normalization_14_314045*
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
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_31351720
.batch_normalization_14/StatefulPartitionedCall�
Relu_1Relu7batch_normalization_14/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_1�
 dense_12/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_12_314057*
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
D__inference_dense_12_layer_call_and_return_conditional_losses_3140562"
 dense_12/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0batch_normalization_15_314060batch_normalization_15_314062batch_normalization_15_314064batch_normalization_15_314066*
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
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_31368320
.batch_normalization_15/StatefulPartitionedCall�
Relu_2Relu7batch_normalization_15/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_2�
 dense_13/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_13_314078*
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
D__inference_dense_13_layer_call_and_return_conditional_losses_3140772"
 dense_13/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0batch_normalization_16_314081batch_normalization_16_314083batch_normalization_16_314085batch_normalization_16_314087*
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
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_31384920
.batch_normalization_16/StatefulPartitionedCall�
Relu_3Relu7batch_normalization_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Relu_3�
 dense_14/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_14_314102dense_14_314104*
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
D__inference_dense_14_layer_call_and_return_conditional_losses_3141012"
 dense_14/StatefulPartitionedCall�
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:J F
'
_output_shapes
:���������


_user_specified_namex
�
�
7__inference_feed_forward_sub_net_2_layer_call_fn_314816
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
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_3143342
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
�
�
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_315585

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
D__inference_dense_10_layer_call_and_return_conditional_losses_315881

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
�
�
7__inference_feed_forward_sub_net_2_layer_call_fn_314759
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
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_3141082
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
�
�
7__inference_batch_normalization_13_layer_call_fn_315565

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
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3134132
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
)__inference_dense_10_layer_call_fn_315874

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
D__inference_dense_10_layer_call_and_return_conditional_losses_3140142
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
�+
�
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_315539

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
�
�
D__inference_dense_13_layer_call_and_return_conditional_losses_315923

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
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_315503

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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
�_default_save_signature
�__call__
+�&call_and_return_all_conditional_losses"
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
 "
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
�
-layer_metrics
regularization_losses
.metrics
	variables
/layer_regularization_losses

0layers
1non_trainable_variables
trainable_variables
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
3regularization_losses
4	variables
5trainable_variables
6	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
7axis
	gamma
beta
moving_mean
 moving_variance
8regularization_losses
9	variables
:trainable_variables
;	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
<axis
	gamma
beta
!moving_mean
"moving_variance
=regularization_losses
>	variables
?trainable_variables
@	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Aaxis
	gamma
beta
#moving_mean
$moving_variance
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Faxis
	gamma
beta
%moving_mean
&moving_variance
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
�

'kernel
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

(kernel
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

)kernel
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

*kernel
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

+kernel
,bias
\regularization_losses
]	variables
^trainable_variables
_	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
A:?
23feed_forward_sub_net_2/batch_normalization_12/gamma
@:>
22feed_forward_sub_net_2/batch_normalization_12/beta
A:?23feed_forward_sub_net_2/batch_normalization_13/gamma
@:>22feed_forward_sub_net_2/batch_normalization_13/beta
A:?23feed_forward_sub_net_2/batch_normalization_14/gamma
@:>22feed_forward_sub_net_2/batch_normalization_14/beta
A:?23feed_forward_sub_net_2/batch_normalization_15/gamma
@:>22feed_forward_sub_net_2/batch_normalization_15/beta
A:?23feed_forward_sub_net_2/batch_normalization_16/gamma
@:>22feed_forward_sub_net_2/batch_normalization_16/beta
I:G
 (29feed_forward_sub_net_2/batch_normalization_12/moving_mean
M:K
 (2=feed_forward_sub_net_2/batch_normalization_12/moving_variance
I:G (29feed_forward_sub_net_2/batch_normalization_13/moving_mean
M:K (2=feed_forward_sub_net_2/batch_normalization_13/moving_variance
I:G (29feed_forward_sub_net_2/batch_normalization_14/moving_mean
M:K (2=feed_forward_sub_net_2/batch_normalization_14/moving_variance
I:G (29feed_forward_sub_net_2/batch_normalization_15/moving_mean
M:K (2=feed_forward_sub_net_2/batch_normalization_15/moving_variance
I:G (29feed_forward_sub_net_2/batch_normalization_16/moving_mean
M:K (2=feed_forward_sub_net_2/batch_normalization_16/moving_variance
8:6
2&feed_forward_sub_net_2/dense_10/kernel
8:62&feed_forward_sub_net_2/dense_11/kernel
8:62&feed_forward_sub_net_2/dense_12/kernel
8:62&feed_forward_sub_net_2/dense_13/kernel
8:6
2&feed_forward_sub_net_2/dense_14/kernel
2:0
2$feed_forward_sub_net_2/dense_14/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
�
`layer_metrics
3regularization_losses
ametrics
4	variables
blayer_regularization_losses

clayers
dnon_trainable_variables
5trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
�
elayer_metrics
8regularization_losses
fmetrics
9	variables
glayer_regularization_losses

hlayers
inon_trainable_variables
:trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
�
jlayer_metrics
=regularization_losses
kmetrics
>	variables
llayer_regularization_losses

mlayers
nnon_trainable_variables
?trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
�
olayer_metrics
Bregularization_losses
pmetrics
C	variables
qlayer_regularization_losses

rlayers
snon_trainable_variables
Dtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
�
tlayer_metrics
Gregularization_losses
umetrics
H	variables
vlayer_regularization_losses

wlayers
xnon_trainable_variables
Itrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
'
'0"
trackable_list_wrapper
'
'0"
trackable_list_wrapper
�
ylayer_metrics
Lregularization_losses
zmetrics
M	variables
{layer_regularization_losses

|layers
}non_trainable_variables
Ntrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
'
(0"
trackable_list_wrapper
�
~layer_metrics
Pregularization_losses
metrics
Q	variables
 �layer_regularization_losses
�layers
�non_trainable_variables
Rtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
�
�layer_metrics
Tregularization_losses
�metrics
U	variables
 �layer_regularization_losses
�layers
�non_trainable_variables
Vtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
�
�layer_metrics
Xregularization_losses
�metrics
Y	variables
 �layer_regularization_losses
�layers
�non_trainable_variables
Ztrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
�
�layer_metrics
\regularization_losses
�metrics
]	variables
 �layer_regularization_losses
�layers
�non_trainable_variables
^trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
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
�B�
!__inference__wrapped_model_313161input_1"�
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
�2�
7__inference_feed_forward_sub_net_2_layer_call_fn_314702
7__inference_feed_forward_sub_net_2_layer_call_fn_314759
7__inference_feed_forward_sub_net_2_layer_call_fn_314816
7__inference_feed_forward_sub_net_2_layer_call_fn_314873�
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
�2�
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_314979
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_315165
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_315271
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_315457�
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
$__inference_signature_wrapper_314645input_1"�
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
7__inference_batch_normalization_12_layer_call_fn_315470
7__inference_batch_normalization_12_layer_call_fn_315483�
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
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_315503
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_315539�
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
7__inference_batch_normalization_13_layer_call_fn_315552
7__inference_batch_normalization_13_layer_call_fn_315565�
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
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_315585
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_315621�
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
7__inference_batch_normalization_14_layer_call_fn_315634
7__inference_batch_normalization_14_layer_call_fn_315647�
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
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_315667
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_315703�
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
7__inference_batch_normalization_15_layer_call_fn_315716
7__inference_batch_normalization_15_layer_call_fn_315729�
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
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_315749
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_315785�
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
7__inference_batch_normalization_16_layer_call_fn_315798
7__inference_batch_normalization_16_layer_call_fn_315811�
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
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_315831
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_315867�
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
)__inference_dense_10_layer_call_fn_315874�
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
D__inference_dense_10_layer_call_and_return_conditional_losses_315881�
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
)__inference_dense_11_layer_call_fn_315888�
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
D__inference_dense_11_layer_call_and_return_conditional_losses_315895�
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
)__inference_dense_12_layer_call_fn_315902�
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
D__inference_dense_12_layer_call_and_return_conditional_losses_315909�
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
)__inference_dense_13_layer_call_fn_315916�
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
D__inference_dense_13_layer_call_and_return_conditional_losses_315923�
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
)__inference_dense_14_layer_call_fn_315932�
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
D__inference_dense_14_layer_call_and_return_conditional_losses_315942�
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
!__inference__wrapped_model_313161�' (!")#$*%&+,0�-
&�#
!�
input_1���������

� "3�0
.
output_1"�
output_1���������
�
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_315503b3�0
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
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_315539b3�0
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
7__inference_batch_normalization_12_layer_call_fn_315470U3�0
)�&
 �
inputs���������

p 
� "����������
�
7__inference_batch_normalization_12_layer_call_fn_315483U3�0
)�&
 �
inputs���������

p
� "����������
�
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_315585b 3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_315621b 3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_13_layer_call_fn_315552U 3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_13_layer_call_fn_315565U 3�0
)�&
 �
inputs���������
p
� "�����������
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_315667b!"3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_14_layer_call_and_return_conditional_losses_315703b!"3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_14_layer_call_fn_315634U!"3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_14_layer_call_fn_315647U!"3�0
)�&
 �
inputs���������
p
� "�����������
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_315749b#$3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_315785b#$3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_15_layer_call_fn_315716U#$3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_15_layer_call_fn_315729U#$3�0
)�&
 �
inputs���������
p
� "�����������
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_315831b%&3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_315867b%&3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
7__inference_batch_normalization_16_layer_call_fn_315798U%&3�0
)�&
 �
inputs���������
p 
� "�����������
7__inference_batch_normalization_16_layer_call_fn_315811U%&3�0
)�&
 �
inputs���������
p
� "�����������
D__inference_dense_10_layer_call_and_return_conditional_losses_315881['/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� {
)__inference_dense_10_layer_call_fn_315874N'/�,
%�"
 �
inputs���������

� "�����������
D__inference_dense_11_layer_call_and_return_conditional_losses_315895[(/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
)__inference_dense_11_layer_call_fn_315888N(/�,
%�"
 �
inputs���������
� "�����������
D__inference_dense_12_layer_call_and_return_conditional_losses_315909[)/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
)__inference_dense_12_layer_call_fn_315902N)/�,
%�"
 �
inputs���������
� "�����������
D__inference_dense_13_layer_call_and_return_conditional_losses_315923[*/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� {
)__inference_dense_13_layer_call_fn_315916N*/�,
%�"
 �
inputs���������
� "�����������
D__inference_dense_14_layer_call_and_return_conditional_losses_315942\+,/�,
%�"
 �
inputs���������
� "%�"
�
0���������

� |
)__inference_dense_14_layer_call_fn_315932O+,/�,
%�"
 �
inputs���������
� "����������
�
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_314979s' (!")#$*%&+,.�+
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
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_315165s' (!")#$*%&+,.�+
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
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_315271y' (!")#$*%&+,4�1
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
R__inference_feed_forward_sub_net_2_layer_call_and_return_conditional_losses_315457y' (!")#$*%&+,4�1
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
7__inference_feed_forward_sub_net_2_layer_call_fn_314702l' (!")#$*%&+,4�1
*�'
!�
input_1���������

p 
� "����������
�
7__inference_feed_forward_sub_net_2_layer_call_fn_314759f' (!")#$*%&+,.�+
$�!
�
x���������

p 
� "����������
�
7__inference_feed_forward_sub_net_2_layer_call_fn_314816f' (!")#$*%&+,.�+
$�!
�
x���������

p
� "����������
�
7__inference_feed_forward_sub_net_2_layer_call_fn_314873l' (!")#$*%&+,4�1
*�'
!�
input_1���������

p
� "����������
�
$__inference_signature_wrapper_314645�' (!")#$*%&+,;�8
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