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
3feed_forward_sub_net_3/batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*D
shared_name53feed_forward_sub_net_3/batch_normalization_18/gamma
·
Gfeed_forward_sub_net_3/batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_3/batch_normalization_18/gamma*
_output_shapes
:
*
dtype0
¼
2feed_forward_sub_net_3/batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*C
shared_name42feed_forward_sub_net_3/batch_normalization_18/beta
µ
Ffeed_forward_sub_net_3/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_3/batch_normalization_18/beta*
_output_shapes
:
*
dtype0
¾
3feed_forward_sub_net_3/batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_3/batch_normalization_19/gamma
·
Gfeed_forward_sub_net_3/batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_3/batch_normalization_19/gamma*
_output_shapes
:*
dtype0
¼
2feed_forward_sub_net_3/batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_3/batch_normalization_19/beta
µ
Ffeed_forward_sub_net_3/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_3/batch_normalization_19/beta*
_output_shapes
:*
dtype0
¾
3feed_forward_sub_net_3/batch_normalization_20/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_3/batch_normalization_20/gamma
·
Gfeed_forward_sub_net_3/batch_normalization_20/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_3/batch_normalization_20/gamma*
_output_shapes
:*
dtype0
¼
2feed_forward_sub_net_3/batch_normalization_20/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_3/batch_normalization_20/beta
µ
Ffeed_forward_sub_net_3/batch_normalization_20/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_3/batch_normalization_20/beta*
_output_shapes
:*
dtype0
¾
3feed_forward_sub_net_3/batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_3/batch_normalization_21/gamma
·
Gfeed_forward_sub_net_3/batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_3/batch_normalization_21/gamma*
_output_shapes
:*
dtype0
¼
2feed_forward_sub_net_3/batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_3/batch_normalization_21/beta
µ
Ffeed_forward_sub_net_3/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_3/batch_normalization_21/beta*
_output_shapes
:*
dtype0
¾
3feed_forward_sub_net_3/batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53feed_forward_sub_net_3/batch_normalization_22/gamma
·
Gfeed_forward_sub_net_3/batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOp3feed_forward_sub_net_3/batch_normalization_22/gamma*
_output_shapes
:*
dtype0
¼
2feed_forward_sub_net_3/batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42feed_forward_sub_net_3/batch_normalization_22/beta
µ
Ffeed_forward_sub_net_3/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOp2feed_forward_sub_net_3/batch_normalization_22/beta*
_output_shapes
:*
dtype0
Ê
9feed_forward_sub_net_3/batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*J
shared_name;9feed_forward_sub_net_3/batch_normalization_18/moving_mean
Ã
Mfeed_forward_sub_net_3/batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_3/batch_normalization_18/moving_mean*
_output_shapes
:
*
dtype0
Ò
=feed_forward_sub_net_3/batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*N
shared_name?=feed_forward_sub_net_3/batch_normalization_18/moving_variance
Ë
Qfeed_forward_sub_net_3/batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_3/batch_normalization_18/moving_variance*
_output_shapes
:
*
dtype0
Ê
9feed_forward_sub_net_3/batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_3/batch_normalization_19/moving_mean
Ã
Mfeed_forward_sub_net_3/batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_3/batch_normalization_19/moving_mean*
_output_shapes
:*
dtype0
Ò
=feed_forward_sub_net_3/batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_3/batch_normalization_19/moving_variance
Ë
Qfeed_forward_sub_net_3/batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_3/batch_normalization_19/moving_variance*
_output_shapes
:*
dtype0
Ê
9feed_forward_sub_net_3/batch_normalization_20/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_3/batch_normalization_20/moving_mean
Ã
Mfeed_forward_sub_net_3/batch_normalization_20/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_3/batch_normalization_20/moving_mean*
_output_shapes
:*
dtype0
Ò
=feed_forward_sub_net_3/batch_normalization_20/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_3/batch_normalization_20/moving_variance
Ë
Qfeed_forward_sub_net_3/batch_normalization_20/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_3/batch_normalization_20/moving_variance*
_output_shapes
:*
dtype0
Ê
9feed_forward_sub_net_3/batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_3/batch_normalization_21/moving_mean
Ã
Mfeed_forward_sub_net_3/batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_3/batch_normalization_21/moving_mean*
_output_shapes
:*
dtype0
Ò
=feed_forward_sub_net_3/batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_3/batch_normalization_21/moving_variance
Ë
Qfeed_forward_sub_net_3/batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_3/batch_normalization_21/moving_variance*
_output_shapes
:*
dtype0
Ê
9feed_forward_sub_net_3/batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9feed_forward_sub_net_3/batch_normalization_22/moving_mean
Ã
Mfeed_forward_sub_net_3/batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp9feed_forward_sub_net_3/batch_normalization_22/moving_mean*
_output_shapes
:*
dtype0
Ò
=feed_forward_sub_net_3/batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=feed_forward_sub_net_3/batch_normalization_22/moving_variance
Ë
Qfeed_forward_sub_net_3/batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp=feed_forward_sub_net_3/batch_normalization_22/moving_variance*
_output_shapes
:*
dtype0
¨
&feed_forward_sub_net_3/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&feed_forward_sub_net_3/dense_15/kernel
¡
:feed_forward_sub_net_3/dense_15/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_3/dense_15/kernel*
_output_shapes

:
*
dtype0
¨
&feed_forward_sub_net_3/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_3/dense_16/kernel
¡
:feed_forward_sub_net_3/dense_16/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_3/dense_16/kernel*
_output_shapes

:*
dtype0
¨
&feed_forward_sub_net_3/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_3/dense_17/kernel
¡
:feed_forward_sub_net_3/dense_17/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_3/dense_17/kernel*
_output_shapes

:*
dtype0
¨
&feed_forward_sub_net_3/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&feed_forward_sub_net_3/dense_18/kernel
¡
:feed_forward_sub_net_3/dense_18/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_3/dense_18/kernel*
_output_shapes

:*
dtype0
¨
&feed_forward_sub_net_3/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&feed_forward_sub_net_3/dense_19/kernel
¡
:feed_forward_sub_net_3/dense_19/kernel/Read/ReadVariableOpReadVariableOp&feed_forward_sub_net_3/dense_19/kernel*
_output_shapes

:
*
dtype0
 
$feed_forward_sub_net_3/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$feed_forward_sub_net_3/dense_19/bias

8feed_forward_sub_net_3/dense_19/bias/Read/ReadVariableOpReadVariableOp$feed_forward_sub_net_3/dense_19/bias*
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
­
-layer_metrics
regularization_losses
.metrics
	variables
/layer_regularization_losses

0layers
1non_trainable_variables
trainable_variables
 

2axis
	gamma
beta
moving_mean
moving_variance
3regularization_losses
4	variables
5trainable_variables
6	keras_api

7axis
	gamma
beta
moving_mean
 moving_variance
8regularization_losses
9	variables
:trainable_variables
;	keras_api

<axis
	gamma
beta
!moving_mean
"moving_variance
=regularization_losses
>	variables
?trainable_variables
@	keras_api

Aaxis
	gamma
beta
#moving_mean
$moving_variance
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api

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
VARIABLE_VALUE3feed_forward_sub_net_3/batch_normalization_18/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_3/batch_normalization_18/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_3/batch_normalization_19/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_3/batch_normalization_19/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_3/batch_normalization_20/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_3/batch_normalization_20/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_3/batch_normalization_21/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_3/batch_normalization_21/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3feed_forward_sub_net_3/batch_normalization_22/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2feed_forward_sub_net_3/batch_normalization_22/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_3/batch_normalization_18/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_3/batch_normalization_18/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_3/batch_normalization_19/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_3/batch_normalization_19/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_3/batch_normalization_20/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_3/batch_normalization_20/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_3/batch_normalization_21/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_3/batch_normalization_21/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9feed_forward_sub_net_3/batch_normalization_22/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE=feed_forward_sub_net_3/batch_normalization_22/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_3/dense_15/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_3/dense_16/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_3/dense_17/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_3/dense_18/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feed_forward_sub_net_3/dense_19/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$feed_forward_sub_net_3/dense_19/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
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
­
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
­
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
­
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
­
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
­
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
­
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
°
~layer_metrics
Pregularization_losses
metrics
Q	variables
 layer_regularization_losses
layers
non_trainable_variables
Rtrainable_variables
 

)0

)0
²
layer_metrics
Tregularization_losses
metrics
U	variables
 layer_regularization_losses
layers
non_trainable_variables
Vtrainable_variables
 

*0

*0
²
layer_metrics
Xregularization_losses
metrics
Y	variables
 layer_regularization_losses
layers
non_trainable_variables
Ztrainable_variables
 

+0
,1

+0
,1
²
layer_metrics
\regularization_losses
metrics
]	variables
 layer_regularization_losses
layers
non_trainable_variables
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
:ÿÿÿÿÿÿÿÿÿ
*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

Ã
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_19feed_forward_sub_net_3/batch_normalization_18/moving_mean=feed_forward_sub_net_3/batch_normalization_18/moving_variance2feed_forward_sub_net_3/batch_normalization_18/beta3feed_forward_sub_net_3/batch_normalization_18/gamma&feed_forward_sub_net_3/dense_15/kernel9feed_forward_sub_net_3/batch_normalization_19/moving_mean=feed_forward_sub_net_3/batch_normalization_19/moving_variance2feed_forward_sub_net_3/batch_normalization_19/beta3feed_forward_sub_net_3/batch_normalization_19/gamma&feed_forward_sub_net_3/dense_16/kernel9feed_forward_sub_net_3/batch_normalization_20/moving_mean=feed_forward_sub_net_3/batch_normalization_20/moving_variance2feed_forward_sub_net_3/batch_normalization_20/beta3feed_forward_sub_net_3/batch_normalization_20/gamma&feed_forward_sub_net_3/dense_17/kernel9feed_forward_sub_net_3/batch_normalization_21/moving_mean=feed_forward_sub_net_3/batch_normalization_21/moving_variance2feed_forward_sub_net_3/batch_normalization_21/beta3feed_forward_sub_net_3/batch_normalization_21/gamma&feed_forward_sub_net_3/dense_18/kernel9feed_forward_sub_net_3/batch_normalization_22/moving_mean=feed_forward_sub_net_3/batch_normalization_22/moving_variance2feed_forward_sub_net_3/batch_normalization_22/beta3feed_forward_sub_net_3/batch_normalization_22/gamma&feed_forward_sub_net_3/dense_19/kernel$feed_forward_sub_net_3/dense_19/bias*&
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
$__inference_signature_wrapper_317841
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameGfeed_forward_sub_net_3/batch_normalization_18/gamma/Read/ReadVariableOpFfeed_forward_sub_net_3/batch_normalization_18/beta/Read/ReadVariableOpGfeed_forward_sub_net_3/batch_normalization_19/gamma/Read/ReadVariableOpFfeed_forward_sub_net_3/batch_normalization_19/beta/Read/ReadVariableOpGfeed_forward_sub_net_3/batch_normalization_20/gamma/Read/ReadVariableOpFfeed_forward_sub_net_3/batch_normalization_20/beta/Read/ReadVariableOpGfeed_forward_sub_net_3/batch_normalization_21/gamma/Read/ReadVariableOpFfeed_forward_sub_net_3/batch_normalization_21/beta/Read/ReadVariableOpGfeed_forward_sub_net_3/batch_normalization_22/gamma/Read/ReadVariableOpFfeed_forward_sub_net_3/batch_normalization_22/beta/Read/ReadVariableOpMfeed_forward_sub_net_3/batch_normalization_18/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_3/batch_normalization_18/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_3/batch_normalization_19/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_3/batch_normalization_19/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_3/batch_normalization_20/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_3/batch_normalization_20/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_3/batch_normalization_21/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_3/batch_normalization_21/moving_variance/Read/ReadVariableOpMfeed_forward_sub_net_3/batch_normalization_22/moving_mean/Read/ReadVariableOpQfeed_forward_sub_net_3/batch_normalization_22/moving_variance/Read/ReadVariableOp:feed_forward_sub_net_3/dense_15/kernel/Read/ReadVariableOp:feed_forward_sub_net_3/dense_16/kernel/Read/ReadVariableOp:feed_forward_sub_net_3/dense_17/kernel/Read/ReadVariableOp:feed_forward_sub_net_3/dense_18/kernel/Read/ReadVariableOp:feed_forward_sub_net_3/dense_19/kernel/Read/ReadVariableOp8feed_forward_sub_net_3/dense_19/bias/Read/ReadVariableOpConst*'
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
__inference__traced_save_319239

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename3feed_forward_sub_net_3/batch_normalization_18/gamma2feed_forward_sub_net_3/batch_normalization_18/beta3feed_forward_sub_net_3/batch_normalization_19/gamma2feed_forward_sub_net_3/batch_normalization_19/beta3feed_forward_sub_net_3/batch_normalization_20/gamma2feed_forward_sub_net_3/batch_normalization_20/beta3feed_forward_sub_net_3/batch_normalization_21/gamma2feed_forward_sub_net_3/batch_normalization_21/beta3feed_forward_sub_net_3/batch_normalization_22/gamma2feed_forward_sub_net_3/batch_normalization_22/beta9feed_forward_sub_net_3/batch_normalization_18/moving_mean=feed_forward_sub_net_3/batch_normalization_18/moving_variance9feed_forward_sub_net_3/batch_normalization_19/moving_mean=feed_forward_sub_net_3/batch_normalization_19/moving_variance9feed_forward_sub_net_3/batch_normalization_20/moving_mean=feed_forward_sub_net_3/batch_normalization_20/moving_variance9feed_forward_sub_net_3/batch_normalization_21/moving_mean=feed_forward_sub_net_3/batch_normalization_21/moving_variance9feed_forward_sub_net_3/batch_normalization_22/moving_mean=feed_forward_sub_net_3/batch_normalization_22/moving_variance&feed_forward_sub_net_3/dense_15/kernel&feed_forward_sub_net_3/dense_16/kernel&feed_forward_sub_net_3/dense_17/kernel&feed_forward_sub_net_3/dense_18/kernel&feed_forward_sub_net_3/dense_19/kernel$feed_forward_sub_net_3/dense_19/bias*&
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
"__inference__traced_restore_319327à
±

R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_318699

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
þ
­
D__inference_dense_16_layer_call_and_return_conditional_losses_319091

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
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_316713

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
D__inference_dense_18_layer_call_and_return_conditional_losses_317273

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
D__inference_dense_15_layer_call_and_return_conditional_losses_317210

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
º

$__inference_signature_wrapper_317841
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
!__inference__wrapped_model_3163572
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
Ù
Ò
7__inference_batch_normalization_21_layer_call_fn_318912

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
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_3168792
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
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_316941

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
Ù
Ò
7__inference_batch_normalization_20_layer_call_fn_318830

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
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_3167132
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
D__inference_dense_16_layer_call_and_return_conditional_losses_317231

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
D__inference_dense_17_layer_call_and_return_conditional_losses_317252

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
)__inference_dense_17_layer_call_fn_319098

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
D__inference_dense_17_layer_call_and_return_conditional_losses_3172522
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
Óz
Ò
"__inference__traced_restore_319327
file_prefixR
Dassignvariableop_feed_forward_sub_net_3_batch_normalization_18_gamma:
S
Eassignvariableop_1_feed_forward_sub_net_3_batch_normalization_18_beta:
T
Fassignvariableop_2_feed_forward_sub_net_3_batch_normalization_19_gamma:S
Eassignvariableop_3_feed_forward_sub_net_3_batch_normalization_19_beta:T
Fassignvariableop_4_feed_forward_sub_net_3_batch_normalization_20_gamma:S
Eassignvariableop_5_feed_forward_sub_net_3_batch_normalization_20_beta:T
Fassignvariableop_6_feed_forward_sub_net_3_batch_normalization_21_gamma:S
Eassignvariableop_7_feed_forward_sub_net_3_batch_normalization_21_beta:T
Fassignvariableop_8_feed_forward_sub_net_3_batch_normalization_22_gamma:S
Eassignvariableop_9_feed_forward_sub_net_3_batch_normalization_22_beta:[
Massignvariableop_10_feed_forward_sub_net_3_batch_normalization_18_moving_mean:
_
Qassignvariableop_11_feed_forward_sub_net_3_batch_normalization_18_moving_variance:
[
Massignvariableop_12_feed_forward_sub_net_3_batch_normalization_19_moving_mean:_
Qassignvariableop_13_feed_forward_sub_net_3_batch_normalization_19_moving_variance:[
Massignvariableop_14_feed_forward_sub_net_3_batch_normalization_20_moving_mean:_
Qassignvariableop_15_feed_forward_sub_net_3_batch_normalization_20_moving_variance:[
Massignvariableop_16_feed_forward_sub_net_3_batch_normalization_21_moving_mean:_
Qassignvariableop_17_feed_forward_sub_net_3_batch_normalization_21_moving_variance:[
Massignvariableop_18_feed_forward_sub_net_3_batch_normalization_22_moving_mean:_
Qassignvariableop_19_feed_forward_sub_net_3_batch_normalization_22_moving_variance:L
:assignvariableop_20_feed_forward_sub_net_3_dense_15_kernel:
L
:assignvariableop_21_feed_forward_sub_net_3_dense_16_kernel:L
:assignvariableop_22_feed_forward_sub_net_3_dense_17_kernel:L
:assignvariableop_23_feed_forward_sub_net_3_dense_18_kernel:L
:assignvariableop_24_feed_forward_sub_net_3_dense_19_kernel:
F
8assignvariableop_25_feed_forward_sub_net_3_dense_19_bias:
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
AssignVariableOpAssignVariableOpDassignvariableop_feed_forward_sub_net_3_batch_normalization_18_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ê
AssignVariableOp_1AssignVariableOpEassignvariableop_1_feed_forward_sub_net_3_batch_normalization_18_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ë
AssignVariableOp_2AssignVariableOpFassignvariableop_2_feed_forward_sub_net_3_batch_normalization_19_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ê
AssignVariableOp_3AssignVariableOpEassignvariableop_3_feed_forward_sub_net_3_batch_normalization_19_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ë
AssignVariableOp_4AssignVariableOpFassignvariableop_4_feed_forward_sub_net_3_batch_normalization_20_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ê
AssignVariableOp_5AssignVariableOpEassignvariableop_5_feed_forward_sub_net_3_batch_normalization_20_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ë
AssignVariableOp_6AssignVariableOpFassignvariableop_6_feed_forward_sub_net_3_batch_normalization_21_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ê
AssignVariableOp_7AssignVariableOpEassignvariableop_7_feed_forward_sub_net_3_batch_normalization_21_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ë
AssignVariableOp_8AssignVariableOpFassignvariableop_8_feed_forward_sub_net_3_batch_normalization_22_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ê
AssignVariableOp_9AssignVariableOpEassignvariableop_9_feed_forward_sub_net_3_batch_normalization_22_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Õ
AssignVariableOp_10AssignVariableOpMassignvariableop_10_feed_forward_sub_net_3_batch_normalization_18_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ù
AssignVariableOp_11AssignVariableOpQassignvariableop_11_feed_forward_sub_net_3_batch_normalization_18_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Õ
AssignVariableOp_12AssignVariableOpMassignvariableop_12_feed_forward_sub_net_3_batch_normalization_19_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ù
AssignVariableOp_13AssignVariableOpQassignvariableop_13_feed_forward_sub_net_3_batch_normalization_19_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Õ
AssignVariableOp_14AssignVariableOpMassignvariableop_14_feed_forward_sub_net_3_batch_normalization_20_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ù
AssignVariableOp_15AssignVariableOpQassignvariableop_15_feed_forward_sub_net_3_batch_normalization_20_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Õ
AssignVariableOp_16AssignVariableOpMassignvariableop_16_feed_forward_sub_net_3_batch_normalization_21_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ù
AssignVariableOp_17AssignVariableOpQassignvariableop_17_feed_forward_sub_net_3_batch_normalization_21_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Õ
AssignVariableOp_18AssignVariableOpMassignvariableop_18_feed_forward_sub_net_3_batch_normalization_22_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ù
AssignVariableOp_19AssignVariableOpQassignvariableop_19_feed_forward_sub_net_3_batch_normalization_22_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Â
AssignVariableOp_20AssignVariableOp:assignvariableop_20_feed_forward_sub_net_3_dense_15_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Â
AssignVariableOp_21AssignVariableOp:assignvariableop_21_feed_forward_sub_net_3_dense_16_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Â
AssignVariableOp_22AssignVariableOp:assignvariableop_22_feed_forward_sub_net_3_dense_17_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Â
AssignVariableOp_23AssignVariableOp:assignvariableop_23_feed_forward_sub_net_3_dense_18_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Â
AssignVariableOp_24AssignVariableOp:assignvariableop_24_feed_forward_sub_net_3_dense_19_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25À
AssignVariableOp_25AssignVariableOp8assignvariableop_25_feed_forward_sub_net_3_dense_19_biasIdentity_25:output:0"/device:CPU:0*
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
ñ

)__inference_dense_19_layer_call_fn_319128

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
D__inference_dense_19_layer_call_and_return_conditional_losses_3172972
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
×
Ò
7__inference_batch_normalization_18_layer_call_fn_318679

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
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_3164432
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
ï
 
!__inference__wrapped_model_316357
input_1X
Jfeed_forward_sub_net_3_batch_normalization_18_cast_readvariableop_resource:
Z
Lfeed_forward_sub_net_3_batch_normalization_18_cast_1_readvariableop_resource:
Z
Lfeed_forward_sub_net_3_batch_normalization_18_cast_2_readvariableop_resource:
Z
Lfeed_forward_sub_net_3_batch_normalization_18_cast_3_readvariableop_resource:
P
>feed_forward_sub_net_3_dense_15_matmul_readvariableop_resource:
X
Jfeed_forward_sub_net_3_batch_normalization_19_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_3_batch_normalization_19_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_3_batch_normalization_19_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_3_batch_normalization_19_cast_3_readvariableop_resource:P
>feed_forward_sub_net_3_dense_16_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_3_batch_normalization_20_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_3_batch_normalization_20_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_3_batch_normalization_20_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_3_batch_normalization_20_cast_3_readvariableop_resource:P
>feed_forward_sub_net_3_dense_17_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_3_batch_normalization_21_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_3_batch_normalization_21_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_3_batch_normalization_21_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_3_batch_normalization_21_cast_3_readvariableop_resource:P
>feed_forward_sub_net_3_dense_18_matmul_readvariableop_resource:X
Jfeed_forward_sub_net_3_batch_normalization_22_cast_readvariableop_resource:Z
Lfeed_forward_sub_net_3_batch_normalization_22_cast_1_readvariableop_resource:Z
Lfeed_forward_sub_net_3_batch_normalization_22_cast_2_readvariableop_resource:Z
Lfeed_forward_sub_net_3_batch_normalization_22_cast_3_readvariableop_resource:P
>feed_forward_sub_net_3_dense_19_matmul_readvariableop_resource:
M
?feed_forward_sub_net_3_dense_19_biasadd_readvariableop_resource:

identity¢Afeed_forward_sub_net_3/batch_normalization_18/Cast/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_18/Cast_1/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_18/Cast_2/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_18/Cast_3/ReadVariableOp¢Afeed_forward_sub_net_3/batch_normalization_19/Cast/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_19/Cast_1/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_19/Cast_2/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_19/Cast_3/ReadVariableOp¢Afeed_forward_sub_net_3/batch_normalization_20/Cast/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_20/Cast_1/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_20/Cast_2/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_20/Cast_3/ReadVariableOp¢Afeed_forward_sub_net_3/batch_normalization_21/Cast/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_21/Cast_1/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_21/Cast_2/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_21/Cast_3/ReadVariableOp¢Afeed_forward_sub_net_3/batch_normalization_22/Cast/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_22/Cast_1/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_22/Cast_2/ReadVariableOp¢Cfeed_forward_sub_net_3/batch_normalization_22/Cast_3/ReadVariableOp¢5feed_forward_sub_net_3/dense_15/MatMul/ReadVariableOp¢5feed_forward_sub_net_3/dense_16/MatMul/ReadVariableOp¢5feed_forward_sub_net_3/dense_17/MatMul/ReadVariableOp¢5feed_forward_sub_net_3/dense_18/MatMul/ReadVariableOp¢6feed_forward_sub_net_3/dense_19/BiasAdd/ReadVariableOp¢5feed_forward_sub_net_3/dense_19/MatMul/ReadVariableOp
Afeed_forward_sub_net_3/batch_normalization_18/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_3_batch_normalization_18_cast_readvariableop_resource*
_output_shapes
:
*
dtype02C
Afeed_forward_sub_net_3/batch_normalization_18/Cast/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_18/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_18_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_18/Cast_1/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_18/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_18_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_18/Cast_2/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_18/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_18_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_18/Cast_3/ReadVariableOpÇ
=feed_forward_sub_net_3/batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2?
=feed_forward_sub_net_3/batch_normalization_18/batchnorm/add/y½
;feed_forward_sub_net_3/batch_normalization_18/batchnorm/addAddV2Kfeed_forward_sub_net_3/batch_normalization_18/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_3/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_3/batch_normalization_18/batchnorm/addí
=feed_forward_sub_net_3/batch_normalization_18/batchnorm/RsqrtRsqrt?feed_forward_sub_net_3/batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_3/batch_normalization_18/batchnorm/Rsqrt¶
;feed_forward_sub_net_3/batch_normalization_18/batchnorm/mulMulAfeed_forward_sub_net_3/batch_normalization_18/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_3/batch_normalization_18/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_3/batch_normalization_18/batchnorm/mul
=feed_forward_sub_net_3/batch_normalization_18/batchnorm/mul_1Mulinput_1?feed_forward_sub_net_3/batch_normalization_18/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2?
=feed_forward_sub_net_3/batch_normalization_18/batchnorm/mul_1¶
=feed_forward_sub_net_3/batch_normalization_18/batchnorm/mul_2MulIfeed_forward_sub_net_3/batch_normalization_18/Cast/ReadVariableOp:value:0?feed_forward_sub_net_3/batch_normalization_18/batchnorm/mul:z:0*
T0*
_output_shapes
:
2?
=feed_forward_sub_net_3/batch_normalization_18/batchnorm/mul_2¶
;feed_forward_sub_net_3/batch_normalization_18/batchnorm/subSubKfeed_forward_sub_net_3/batch_normalization_18/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_3/batch_normalization_18/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2=
;feed_forward_sub_net_3/batch_normalization_18/batchnorm/sub½
=feed_forward_sub_net_3/batch_normalization_18/batchnorm/add_1AddV2Afeed_forward_sub_net_3/batch_normalization_18/batchnorm/mul_1:z:0?feed_forward_sub_net_3/batch_normalization_18/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2?
=feed_forward_sub_net_3/batch_normalization_18/batchnorm/add_1í
5feed_forward_sub_net_3/dense_15/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_3_dense_15_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5feed_forward_sub_net_3/dense_15/MatMul/ReadVariableOp
&feed_forward_sub_net_3/dense_15/MatMulMatMulAfeed_forward_sub_net_3/batch_normalization_18/batchnorm/add_1:z:0=feed_forward_sub_net_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&feed_forward_sub_net_3/dense_15/MatMul
Afeed_forward_sub_net_3/batch_normalization_19/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_3_batch_normalization_19_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_3/batch_normalization_19/Cast/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_19/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_19_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_19/Cast_1/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_19/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_19_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_19/Cast_2/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_19/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_19_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_19/Cast_3/ReadVariableOpÇ
=feed_forward_sub_net_3/batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2?
=feed_forward_sub_net_3/batch_normalization_19/batchnorm/add/y½
;feed_forward_sub_net_3/batch_normalization_19/batchnorm/addAddV2Kfeed_forward_sub_net_3/batch_normalization_19/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_3/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_3/batch_normalization_19/batchnorm/addí
=feed_forward_sub_net_3/batch_normalization_19/batchnorm/RsqrtRsqrt?feed_forward_sub_net_3/batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_3/batch_normalization_19/batchnorm/Rsqrt¶
;feed_forward_sub_net_3/batch_normalization_19/batchnorm/mulMulAfeed_forward_sub_net_3/batch_normalization_19/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_3/batch_normalization_19/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_3/batch_normalization_19/batchnorm/mulª
=feed_forward_sub_net_3/batch_normalization_19/batchnorm/mul_1Mul0feed_forward_sub_net_3/dense_15/MatMul:product:0?feed_forward_sub_net_3/batch_normalization_19/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_3/batch_normalization_19/batchnorm/mul_1¶
=feed_forward_sub_net_3/batch_normalization_19/batchnorm/mul_2MulIfeed_forward_sub_net_3/batch_normalization_19/Cast/ReadVariableOp:value:0?feed_forward_sub_net_3/batch_normalization_19/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_3/batch_normalization_19/batchnorm/mul_2¶
;feed_forward_sub_net_3/batch_normalization_19/batchnorm/subSubKfeed_forward_sub_net_3/batch_normalization_19/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_3/batch_normalization_19/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_3/batch_normalization_19/batchnorm/sub½
=feed_forward_sub_net_3/batch_normalization_19/batchnorm/add_1AddV2Afeed_forward_sub_net_3/batch_normalization_19/batchnorm/mul_1:z:0?feed_forward_sub_net_3/batch_normalization_19/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_3/batch_normalization_19/batchnorm/add_1·
feed_forward_sub_net_3/ReluReluAfeed_forward_sub_net_3/batch_normalization_19/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
feed_forward_sub_net_3/Reluí
5feed_forward_sub_net_3/dense_16/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_3_dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_3/dense_16/MatMul/ReadVariableOpö
&feed_forward_sub_net_3/dense_16/MatMulMatMul)feed_forward_sub_net_3/Relu:activations:0=feed_forward_sub_net_3/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&feed_forward_sub_net_3/dense_16/MatMul
Afeed_forward_sub_net_3/batch_normalization_20/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_3_batch_normalization_20_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_3/batch_normalization_20/Cast/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_20/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_20_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_20/Cast_1/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_20/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_20_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_20/Cast_2/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_20/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_20_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_20/Cast_3/ReadVariableOpÇ
=feed_forward_sub_net_3/batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2?
=feed_forward_sub_net_3/batch_normalization_20/batchnorm/add/y½
;feed_forward_sub_net_3/batch_normalization_20/batchnorm/addAddV2Kfeed_forward_sub_net_3/batch_normalization_20/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_3/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_3/batch_normalization_20/batchnorm/addí
=feed_forward_sub_net_3/batch_normalization_20/batchnorm/RsqrtRsqrt?feed_forward_sub_net_3/batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_3/batch_normalization_20/batchnorm/Rsqrt¶
;feed_forward_sub_net_3/batch_normalization_20/batchnorm/mulMulAfeed_forward_sub_net_3/batch_normalization_20/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_3/batch_normalization_20/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_3/batch_normalization_20/batchnorm/mulª
=feed_forward_sub_net_3/batch_normalization_20/batchnorm/mul_1Mul0feed_forward_sub_net_3/dense_16/MatMul:product:0?feed_forward_sub_net_3/batch_normalization_20/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_3/batch_normalization_20/batchnorm/mul_1¶
=feed_forward_sub_net_3/batch_normalization_20/batchnorm/mul_2MulIfeed_forward_sub_net_3/batch_normalization_20/Cast/ReadVariableOp:value:0?feed_forward_sub_net_3/batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_3/batch_normalization_20/batchnorm/mul_2¶
;feed_forward_sub_net_3/batch_normalization_20/batchnorm/subSubKfeed_forward_sub_net_3/batch_normalization_20/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_3/batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_3/batch_normalization_20/batchnorm/sub½
=feed_forward_sub_net_3/batch_normalization_20/batchnorm/add_1AddV2Afeed_forward_sub_net_3/batch_normalization_20/batchnorm/mul_1:z:0?feed_forward_sub_net_3/batch_normalization_20/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_3/batch_normalization_20/batchnorm/add_1»
feed_forward_sub_net_3/Relu_1ReluAfeed_forward_sub_net_3/batch_normalization_20/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
feed_forward_sub_net_3/Relu_1í
5feed_forward_sub_net_3/dense_17/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_3_dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_3/dense_17/MatMul/ReadVariableOpø
&feed_forward_sub_net_3/dense_17/MatMulMatMul+feed_forward_sub_net_3/Relu_1:activations:0=feed_forward_sub_net_3/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&feed_forward_sub_net_3/dense_17/MatMul
Afeed_forward_sub_net_3/batch_normalization_21/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_3_batch_normalization_21_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_3/batch_normalization_21/Cast/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_21/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_21_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_21/Cast_1/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_21/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_21_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_21/Cast_2/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_21/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_21_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_21/Cast_3/ReadVariableOpÇ
=feed_forward_sub_net_3/batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2?
=feed_forward_sub_net_3/batch_normalization_21/batchnorm/add/y½
;feed_forward_sub_net_3/batch_normalization_21/batchnorm/addAddV2Kfeed_forward_sub_net_3/batch_normalization_21/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_3/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_3/batch_normalization_21/batchnorm/addí
=feed_forward_sub_net_3/batch_normalization_21/batchnorm/RsqrtRsqrt?feed_forward_sub_net_3/batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_3/batch_normalization_21/batchnorm/Rsqrt¶
;feed_forward_sub_net_3/batch_normalization_21/batchnorm/mulMulAfeed_forward_sub_net_3/batch_normalization_21/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_3/batch_normalization_21/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_3/batch_normalization_21/batchnorm/mulª
=feed_forward_sub_net_3/batch_normalization_21/batchnorm/mul_1Mul0feed_forward_sub_net_3/dense_17/MatMul:product:0?feed_forward_sub_net_3/batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_3/batch_normalization_21/batchnorm/mul_1¶
=feed_forward_sub_net_3/batch_normalization_21/batchnorm/mul_2MulIfeed_forward_sub_net_3/batch_normalization_21/Cast/ReadVariableOp:value:0?feed_forward_sub_net_3/batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_3/batch_normalization_21/batchnorm/mul_2¶
;feed_forward_sub_net_3/batch_normalization_21/batchnorm/subSubKfeed_forward_sub_net_3/batch_normalization_21/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_3/batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_3/batch_normalization_21/batchnorm/sub½
=feed_forward_sub_net_3/batch_normalization_21/batchnorm/add_1AddV2Afeed_forward_sub_net_3/batch_normalization_21/batchnorm/mul_1:z:0?feed_forward_sub_net_3/batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_3/batch_normalization_21/batchnorm/add_1»
feed_forward_sub_net_3/Relu_2ReluAfeed_forward_sub_net_3/batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
feed_forward_sub_net_3/Relu_2í
5feed_forward_sub_net_3/dense_18/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_3_dense_18_matmul_readvariableop_resource*
_output_shapes

:*
dtype027
5feed_forward_sub_net_3/dense_18/MatMul/ReadVariableOpø
&feed_forward_sub_net_3/dense_18/MatMulMatMul+feed_forward_sub_net_3/Relu_2:activations:0=feed_forward_sub_net_3/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&feed_forward_sub_net_3/dense_18/MatMul
Afeed_forward_sub_net_3/batch_normalization_22/Cast/ReadVariableOpReadVariableOpJfeed_forward_sub_net_3_batch_normalization_22_cast_readvariableop_resource*
_output_shapes
:*
dtype02C
Afeed_forward_sub_net_3/batch_normalization_22/Cast/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_22/Cast_1/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_22_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_22/Cast_1/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_22/Cast_2/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_22_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_22/Cast_2/ReadVariableOp
Cfeed_forward_sub_net_3/batch_normalization_22/Cast_3/ReadVariableOpReadVariableOpLfeed_forward_sub_net_3_batch_normalization_22_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02E
Cfeed_forward_sub_net_3/batch_normalization_22/Cast_3/ReadVariableOpÇ
=feed_forward_sub_net_3/batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2?
=feed_forward_sub_net_3/batch_normalization_22/batchnorm/add/y½
;feed_forward_sub_net_3/batch_normalization_22/batchnorm/addAddV2Kfeed_forward_sub_net_3/batch_normalization_22/Cast_1/ReadVariableOp:value:0Ffeed_forward_sub_net_3/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_3/batch_normalization_22/batchnorm/addí
=feed_forward_sub_net_3/batch_normalization_22/batchnorm/RsqrtRsqrt?feed_forward_sub_net_3/batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_3/batch_normalization_22/batchnorm/Rsqrt¶
;feed_forward_sub_net_3/batch_normalization_22/batchnorm/mulMulAfeed_forward_sub_net_3/batch_normalization_22/batchnorm/Rsqrt:y:0Kfeed_forward_sub_net_3/batch_normalization_22/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_3/batch_normalization_22/batchnorm/mulª
=feed_forward_sub_net_3/batch_normalization_22/batchnorm/mul_1Mul0feed_forward_sub_net_3/dense_18/MatMul:product:0?feed_forward_sub_net_3/batch_normalization_22/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_3/batch_normalization_22/batchnorm/mul_1¶
=feed_forward_sub_net_3/batch_normalization_22/batchnorm/mul_2MulIfeed_forward_sub_net_3/batch_normalization_22/Cast/ReadVariableOp:value:0?feed_forward_sub_net_3/batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes
:2?
=feed_forward_sub_net_3/batch_normalization_22/batchnorm/mul_2¶
;feed_forward_sub_net_3/batch_normalization_22/batchnorm/subSubKfeed_forward_sub_net_3/batch_normalization_22/Cast_2/ReadVariableOp:value:0Afeed_forward_sub_net_3/batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2=
;feed_forward_sub_net_3/batch_normalization_22/batchnorm/sub½
=feed_forward_sub_net_3/batch_normalization_22/batchnorm/add_1AddV2Afeed_forward_sub_net_3/batch_normalization_22/batchnorm/mul_1:z:0?feed_forward_sub_net_3/batch_normalization_22/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=feed_forward_sub_net_3/batch_normalization_22/batchnorm/add_1»
feed_forward_sub_net_3/Relu_3ReluAfeed_forward_sub_net_3/batch_normalization_22/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
feed_forward_sub_net_3/Relu_3í
5feed_forward_sub_net_3/dense_19/MatMul/ReadVariableOpReadVariableOp>feed_forward_sub_net_3_dense_19_matmul_readvariableop_resource*
_output_shapes

:
*
dtype027
5feed_forward_sub_net_3/dense_19/MatMul/ReadVariableOpø
&feed_forward_sub_net_3/dense_19/MatMulMatMul+feed_forward_sub_net_3/Relu_3:activations:0=feed_forward_sub_net_3/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&feed_forward_sub_net_3/dense_19/MatMulì
6feed_forward_sub_net_3/dense_19/BiasAdd/ReadVariableOpReadVariableOp?feed_forward_sub_net_3_dense_19_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype028
6feed_forward_sub_net_3/dense_19/BiasAdd/ReadVariableOp
'feed_forward_sub_net_3/dense_19/BiasAddBiasAdd0feed_forward_sub_net_3/dense_19/MatMul:product:0>feed_forward_sub_net_3/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2)
'feed_forward_sub_net_3/dense_19/BiasAdd
IdentityIdentity0feed_forward_sub_net_3/dense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity
NoOpNoOpB^feed_forward_sub_net_3/batch_normalization_18/Cast/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_18/Cast_1/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_18/Cast_2/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_18/Cast_3/ReadVariableOpB^feed_forward_sub_net_3/batch_normalization_19/Cast/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_19/Cast_1/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_19/Cast_2/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_19/Cast_3/ReadVariableOpB^feed_forward_sub_net_3/batch_normalization_20/Cast/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_20/Cast_1/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_20/Cast_2/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_20/Cast_3/ReadVariableOpB^feed_forward_sub_net_3/batch_normalization_21/Cast/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_21/Cast_1/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_21/Cast_2/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_21/Cast_3/ReadVariableOpB^feed_forward_sub_net_3/batch_normalization_22/Cast/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_22/Cast_1/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_22/Cast_2/ReadVariableOpD^feed_forward_sub_net_3/batch_normalization_22/Cast_3/ReadVariableOp6^feed_forward_sub_net_3/dense_15/MatMul/ReadVariableOp6^feed_forward_sub_net_3/dense_16/MatMul/ReadVariableOp6^feed_forward_sub_net_3/dense_17/MatMul/ReadVariableOp6^feed_forward_sub_net_3/dense_18/MatMul/ReadVariableOp7^feed_forward_sub_net_3/dense_19/BiasAdd/ReadVariableOp6^feed_forward_sub_net_3/dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2
Afeed_forward_sub_net_3/batch_normalization_18/Cast/ReadVariableOpAfeed_forward_sub_net_3/batch_normalization_18/Cast/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_18/Cast_1/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_18/Cast_1/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_18/Cast_2/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_18/Cast_2/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_18/Cast_3/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_18/Cast_3/ReadVariableOp2
Afeed_forward_sub_net_3/batch_normalization_19/Cast/ReadVariableOpAfeed_forward_sub_net_3/batch_normalization_19/Cast/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_19/Cast_1/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_19/Cast_1/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_19/Cast_2/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_19/Cast_2/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_19/Cast_3/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_19/Cast_3/ReadVariableOp2
Afeed_forward_sub_net_3/batch_normalization_20/Cast/ReadVariableOpAfeed_forward_sub_net_3/batch_normalization_20/Cast/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_20/Cast_1/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_20/Cast_1/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_20/Cast_2/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_20/Cast_2/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_20/Cast_3/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_20/Cast_3/ReadVariableOp2
Afeed_forward_sub_net_3/batch_normalization_21/Cast/ReadVariableOpAfeed_forward_sub_net_3/batch_normalization_21/Cast/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_21/Cast_1/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_21/Cast_1/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_21/Cast_2/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_21/Cast_2/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_21/Cast_3/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_21/Cast_3/ReadVariableOp2
Afeed_forward_sub_net_3/batch_normalization_22/Cast/ReadVariableOpAfeed_forward_sub_net_3/batch_normalization_22/Cast/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_22/Cast_1/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_22/Cast_1/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_22/Cast_2/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_22/Cast_2/ReadVariableOp2
Cfeed_forward_sub_net_3/batch_normalization_22/Cast_3/ReadVariableOpCfeed_forward_sub_net_3/batch_normalization_22/Cast_3/ReadVariableOp2n
5feed_forward_sub_net_3/dense_15/MatMul/ReadVariableOp5feed_forward_sub_net_3/dense_15/MatMul/ReadVariableOp2n
5feed_forward_sub_net_3/dense_16/MatMul/ReadVariableOp5feed_forward_sub_net_3/dense_16/MatMul/ReadVariableOp2n
5feed_forward_sub_net_3/dense_17/MatMul/ReadVariableOp5feed_forward_sub_net_3/dense_17/MatMul/ReadVariableOp2n
5feed_forward_sub_net_3/dense_18/MatMul/ReadVariableOp5feed_forward_sub_net_3/dense_18/MatMul/ReadVariableOp2p
6feed_forward_sub_net_3/dense_19/BiasAdd/ReadVariableOp6feed_forward_sub_net_3/dense_19/BiasAdd/ReadVariableOp2n
5feed_forward_sub_net_3/dense_19/MatMul/ReadVariableOp5feed_forward_sub_net_3/dense_19/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
Ü

R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_318653
input_1L
>batch_normalization_18_assignmovingavg_readvariableop_resource:
N
@batch_normalization_18_assignmovingavg_1_readvariableop_resource:
A
3batch_normalization_18_cast_readvariableop_resource:
C
5batch_normalization_18_cast_1_readvariableop_resource:
9
'dense_15_matmul_readvariableop_resource:
L
>batch_normalization_19_assignmovingavg_readvariableop_resource:N
@batch_normalization_19_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_19_cast_readvariableop_resource:C
5batch_normalization_19_cast_1_readvariableop_resource:9
'dense_16_matmul_readvariableop_resource:L
>batch_normalization_20_assignmovingavg_readvariableop_resource:N
@batch_normalization_20_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_20_cast_readvariableop_resource:C
5batch_normalization_20_cast_1_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:L
>batch_normalization_21_assignmovingavg_readvariableop_resource:N
@batch_normalization_21_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_21_cast_readvariableop_resource:C
5batch_normalization_21_cast_1_readvariableop_resource:9
'dense_18_matmul_readvariableop_resource:L
>batch_normalization_22_assignmovingavg_readvariableop_resource:N
@batch_normalization_22_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_22_cast_readvariableop_resource:C
5batch_normalization_22_cast_1_readvariableop_resource:9
'dense_19_matmul_readvariableop_resource:
6
(dense_19_biasadd_readvariableop_resource:

identity¢&batch_normalization_18/AssignMovingAvg¢5batch_normalization_18/AssignMovingAvg/ReadVariableOp¢(batch_normalization_18/AssignMovingAvg_1¢7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_18/Cast/ReadVariableOp¢,batch_normalization_18/Cast_1/ReadVariableOp¢&batch_normalization_19/AssignMovingAvg¢5batch_normalization_19/AssignMovingAvg/ReadVariableOp¢(batch_normalization_19/AssignMovingAvg_1¢7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_19/Cast/ReadVariableOp¢,batch_normalization_19/Cast_1/ReadVariableOp¢&batch_normalization_20/AssignMovingAvg¢5batch_normalization_20/AssignMovingAvg/ReadVariableOp¢(batch_normalization_20/AssignMovingAvg_1¢7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_20/Cast/ReadVariableOp¢,batch_normalization_20/Cast_1/ReadVariableOp¢&batch_normalization_21/AssignMovingAvg¢5batch_normalization_21/AssignMovingAvg/ReadVariableOp¢(batch_normalization_21/AssignMovingAvg_1¢7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_21/Cast/ReadVariableOp¢,batch_normalization_21/Cast_1/ReadVariableOp¢&batch_normalization_22/AssignMovingAvg¢5batch_normalization_22/AssignMovingAvg/ReadVariableOp¢(batch_normalization_22/AssignMovingAvg_1¢7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_22/Cast/ReadVariableOp¢,batch_normalization_22/Cast_1/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¸
5batch_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_18/moments/mean/reduction_indicesÕ
#batch_normalization_18/moments/meanMeaninput_1>batch_normalization_18/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_18/moments/meanÁ
+batch_normalization_18/moments/StopGradientStopGradient,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_18/moments/StopGradientê
0batch_normalization_18/moments/SquaredDifferenceSquaredDifferenceinput_14batch_normalization_18/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
22
0batch_normalization_18/moments/SquaredDifferenceÀ
9batch_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_18/moments/variance/reduction_indices
'batch_normalization_18/moments/varianceMean4batch_normalization_18/moments/SquaredDifference:z:0Bbatch_normalization_18/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_18/moments/varianceÅ
&batch_normalization_18/moments/SqueezeSqueeze,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_18/moments/SqueezeÍ
(batch_normalization_18/moments/Squeeze_1Squeeze0batch_normalization_18/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_18/moments/Squeeze_1¡
,batch_normalization_18/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_18/AssignMovingAvg/decayÉ
+batch_normalization_18/AssignMovingAvg/CastCast5batch_normalization_18/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_18/AssignMovingAvg/Casté
5batch_normalization_18/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_18/AssignMovingAvg/ReadVariableOpô
*batch_normalization_18/AssignMovingAvg/subSub=batch_normalization_18/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_18/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_18/AssignMovingAvg/subå
*batch_normalization_18/AssignMovingAvg/mulMul.batch_normalization_18/AssignMovingAvg/sub:z:0/batch_normalization_18/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_18/AssignMovingAvg/mul²
&batch_normalization_18/AssignMovingAvgAssignSubVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource.batch_normalization_18/AssignMovingAvg/mul:z:06^batch_normalization_18/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_18/AssignMovingAvg¥
.batch_normalization_18/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_18/AssignMovingAvg_1/decayÏ
-batch_normalization_18/AssignMovingAvg_1/CastCast7batch_normalization_18/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_18/AssignMovingAvg_1/Castï
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_18/AssignMovingAvg_1/subSub?batch_normalization_18/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_18/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_18/AssignMovingAvg_1/subí
,batch_normalization_18/AssignMovingAvg_1/mulMul0batch_normalization_18/AssignMovingAvg_1/sub:z:01batch_normalization_18/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_18/AssignMovingAvg_1/mul¼
(batch_normalization_18/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource0batch_normalization_18/AssignMovingAvg_1/mul:z:08^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_18/AssignMovingAvg_1È
*batch_normalization_18/Cast/ReadVariableOpReadVariableOp3batch_normalization_18_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_18/Cast/ReadVariableOpÎ
,batch_normalization_18/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_18_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_18/Cast_1/ReadVariableOp
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_18/batchnorm/add/yÞ
$batch_normalization_18/batchnorm/addAddV21batch_normalization_18/moments/Squeeze_1:output:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_18/batchnorm/add¨
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_18/batchnorm/RsqrtÚ
$batch_normalization_18/batchnorm/mulMul*batch_normalization_18/batchnorm/Rsqrt:y:04batch_normalization_18/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_18/batchnorm/mul¼
&batch_normalization_18/batchnorm/mul_1Mulinput_1(batch_normalization_18/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_18/batchnorm/mul_1×
&batch_normalization_18/batchnorm/mul_2Mul/batch_normalization_18/moments/Squeeze:output:0(batch_normalization_18/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_18/batchnorm/mul_2Ø
$batch_normalization_18/batchnorm/subSub2batch_normalization_18/Cast/ReadVariableOp:value:0*batch_normalization_18/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_18/batchnorm/subá
&batch_normalization_18/batchnorm/add_1AddV2*batch_normalization_18/batchnorm/mul_1:z:0(batch_normalization_18/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_18/batchnorm/add_1¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_15/MatMul/ReadVariableOp²
dense_15/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMul¸
5batch_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_19/moments/mean/reduction_indicesç
#batch_normalization_19/moments/meanMeandense_15/MatMul:product:0>batch_normalization_19/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_19/moments/meanÁ
+batch_normalization_19/moments/StopGradientStopGradient,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_19/moments/StopGradientü
0batch_normalization_19/moments/SquaredDifferenceSquaredDifferencedense_15/MatMul:product:04batch_normalization_19/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_19/moments/SquaredDifferenceÀ
9batch_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_19/moments/variance/reduction_indices
'batch_normalization_19/moments/varianceMean4batch_normalization_19/moments/SquaredDifference:z:0Bbatch_normalization_19/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_19/moments/varianceÅ
&batch_normalization_19/moments/SqueezeSqueeze,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_19/moments/SqueezeÍ
(batch_normalization_19/moments/Squeeze_1Squeeze0batch_normalization_19/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_19/moments/Squeeze_1¡
,batch_normalization_19/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_19/AssignMovingAvg/decayÉ
+batch_normalization_19/AssignMovingAvg/CastCast5batch_normalization_19/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_19/AssignMovingAvg/Casté
5batch_normalization_19/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_19_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_19/AssignMovingAvg/ReadVariableOpô
*batch_normalization_19/AssignMovingAvg/subSub=batch_normalization_19/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_19/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_19/AssignMovingAvg/subå
*batch_normalization_19/AssignMovingAvg/mulMul.batch_normalization_19/AssignMovingAvg/sub:z:0/batch_normalization_19/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_19/AssignMovingAvg/mul²
&batch_normalization_19/AssignMovingAvgAssignSubVariableOp>batch_normalization_19_assignmovingavg_readvariableop_resource.batch_normalization_19/AssignMovingAvg/mul:z:06^batch_normalization_19/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_19/AssignMovingAvg¥
.batch_normalization_19/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_19/AssignMovingAvg_1/decayÏ
-batch_normalization_19/AssignMovingAvg_1/CastCast7batch_normalization_19/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_19/AssignMovingAvg_1/Castï
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_19_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_19/AssignMovingAvg_1/subSub?batch_normalization_19/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_19/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_19/AssignMovingAvg_1/subí
,batch_normalization_19/AssignMovingAvg_1/mulMul0batch_normalization_19/AssignMovingAvg_1/sub:z:01batch_normalization_19/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_19/AssignMovingAvg_1/mul¼
(batch_normalization_19/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_19_assignmovingavg_1_readvariableop_resource0batch_normalization_19/AssignMovingAvg_1/mul:z:08^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_19/AssignMovingAvg_1È
*batch_normalization_19/Cast/ReadVariableOpReadVariableOp3batch_normalization_19_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_19/Cast/ReadVariableOpÎ
,batch_normalization_19/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_19_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_19/Cast_1/ReadVariableOp
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_19/batchnorm/add/yÞ
$batch_normalization_19/batchnorm/addAddV21batch_normalization_19/moments/Squeeze_1:output:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/add¨
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_19/batchnorm/RsqrtÚ
$batch_normalization_19/batchnorm/mulMul*batch_normalization_19/batchnorm/Rsqrt:y:04batch_normalization_19/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/mulÎ
&batch_normalization_19/batchnorm/mul_1Muldense_15/MatMul:product:0(batch_normalization_19/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_19/batchnorm/mul_1×
&batch_normalization_19/batchnorm/mul_2Mul/batch_normalization_19/moments/Squeeze:output:0(batch_normalization_19/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_19/batchnorm/mul_2Ø
$batch_normalization_19/batchnorm/subSub2batch_normalization_19/Cast/ReadVariableOp:value:0*batch_normalization_19/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/subá
&batch_normalization_19/batchnorm/add_1AddV2*batch_normalization_19/batchnorm/mul_1:z:0(batch_normalization_19/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_19/batchnorm/add_1r
ReluRelu*batch_normalization_19/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¨
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_16/MatMul/ReadVariableOp
dense_16/MatMulMatMulRelu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/MatMul¸
5batch_normalization_20/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_20/moments/mean/reduction_indicesç
#batch_normalization_20/moments/meanMeandense_16/MatMul:product:0>batch_normalization_20/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_20/moments/meanÁ
+batch_normalization_20/moments/StopGradientStopGradient,batch_normalization_20/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_20/moments/StopGradientü
0batch_normalization_20/moments/SquaredDifferenceSquaredDifferencedense_16/MatMul:product:04batch_normalization_20/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_20/moments/SquaredDifferenceÀ
9batch_normalization_20/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_20/moments/variance/reduction_indices
'batch_normalization_20/moments/varianceMean4batch_normalization_20/moments/SquaredDifference:z:0Bbatch_normalization_20/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_20/moments/varianceÅ
&batch_normalization_20/moments/SqueezeSqueeze,batch_normalization_20/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_20/moments/SqueezeÍ
(batch_normalization_20/moments/Squeeze_1Squeeze0batch_normalization_20/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_20/moments/Squeeze_1¡
,batch_normalization_20/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_20/AssignMovingAvg/decayÉ
+batch_normalization_20/AssignMovingAvg/CastCast5batch_normalization_20/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_20/AssignMovingAvg/Casté
5batch_normalization_20/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_20_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_20/AssignMovingAvg/ReadVariableOpô
*batch_normalization_20/AssignMovingAvg/subSub=batch_normalization_20/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_20/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_20/AssignMovingAvg/subå
*batch_normalization_20/AssignMovingAvg/mulMul.batch_normalization_20/AssignMovingAvg/sub:z:0/batch_normalization_20/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_20/AssignMovingAvg/mul²
&batch_normalization_20/AssignMovingAvgAssignSubVariableOp>batch_normalization_20_assignmovingavg_readvariableop_resource.batch_normalization_20/AssignMovingAvg/mul:z:06^batch_normalization_20/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_20/AssignMovingAvg¥
.batch_normalization_20/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_20/AssignMovingAvg_1/decayÏ
-batch_normalization_20/AssignMovingAvg_1/CastCast7batch_normalization_20/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_20/AssignMovingAvg_1/Castï
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_20_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_20/AssignMovingAvg_1/subSub?batch_normalization_20/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_20/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_20/AssignMovingAvg_1/subí
,batch_normalization_20/AssignMovingAvg_1/mulMul0batch_normalization_20/AssignMovingAvg_1/sub:z:01batch_normalization_20/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_20/AssignMovingAvg_1/mul¼
(batch_normalization_20/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_20_assignmovingavg_1_readvariableop_resource0batch_normalization_20/AssignMovingAvg_1/mul:z:08^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_20/AssignMovingAvg_1È
*batch_normalization_20/Cast/ReadVariableOpReadVariableOp3batch_normalization_20_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_20/Cast/ReadVariableOpÎ
,batch_normalization_20/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_20_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_20/Cast_1/ReadVariableOp
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_20/batchnorm/add/yÞ
$batch_normalization_20/batchnorm/addAddV21batch_normalization_20/moments/Squeeze_1:output:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_20/batchnorm/add¨
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_20/batchnorm/RsqrtÚ
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:04batch_normalization_20/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_20/batchnorm/mulÎ
&batch_normalization_20/batchnorm/mul_1Muldense_16/MatMul:product:0(batch_normalization_20/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_20/batchnorm/mul_1×
&batch_normalization_20/batchnorm/mul_2Mul/batch_normalization_20/moments/Squeeze:output:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_20/batchnorm/mul_2Ø
$batch_normalization_20/batchnorm/subSub2batch_normalization_20/Cast/ReadVariableOp:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_20/batchnorm/subá
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_20/batchnorm/add_1v
Relu_1Relu*batch_normalization_20/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1¨
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_17/MatMul/ReadVariableOp
dense_17/MatMulMatMulRelu_1:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/MatMul¸
5batch_normalization_21/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_21/moments/mean/reduction_indicesç
#batch_normalization_21/moments/meanMeandense_17/MatMul:product:0>batch_normalization_21/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_21/moments/meanÁ
+batch_normalization_21/moments/StopGradientStopGradient,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_21/moments/StopGradientü
0batch_normalization_21/moments/SquaredDifferenceSquaredDifferencedense_17/MatMul:product:04batch_normalization_21/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_21/moments/SquaredDifferenceÀ
9batch_normalization_21/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_21/moments/variance/reduction_indices
'batch_normalization_21/moments/varianceMean4batch_normalization_21/moments/SquaredDifference:z:0Bbatch_normalization_21/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_21/moments/varianceÅ
&batch_normalization_21/moments/SqueezeSqueeze,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_21/moments/SqueezeÍ
(batch_normalization_21/moments/Squeeze_1Squeeze0batch_normalization_21/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_21/moments/Squeeze_1¡
,batch_normalization_21/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_21/AssignMovingAvg/decayÉ
+batch_normalization_21/AssignMovingAvg/CastCast5batch_normalization_21/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_21/AssignMovingAvg/Casté
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_21/AssignMovingAvg/ReadVariableOpô
*batch_normalization_21/AssignMovingAvg/subSub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_21/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_21/AssignMovingAvg/subå
*batch_normalization_21/AssignMovingAvg/mulMul.batch_normalization_21/AssignMovingAvg/sub:z:0/batch_normalization_21/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_21/AssignMovingAvg/mul²
&batch_normalization_21/AssignMovingAvgAssignSubVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_21/AssignMovingAvg¥
.batch_normalization_21/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_21/AssignMovingAvg_1/decayÏ
-batch_normalization_21/AssignMovingAvg_1/CastCast7batch_normalization_21/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_21/AssignMovingAvg_1/Castï
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_21/AssignMovingAvg_1/subSub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_21/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_21/AssignMovingAvg_1/subí
,batch_normalization_21/AssignMovingAvg_1/mulMul0batch_normalization_21/AssignMovingAvg_1/sub:z:01batch_normalization_21/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_21/AssignMovingAvg_1/mul¼
(batch_normalization_21/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource0batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_21/AssignMovingAvg_1È
*batch_normalization_21/Cast/ReadVariableOpReadVariableOp3batch_normalization_21_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_21/Cast/ReadVariableOpÎ
,batch_normalization_21/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_21_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_21/Cast_1/ReadVariableOp
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_21/batchnorm/add/yÞ
$batch_normalization_21/batchnorm/addAddV21batch_normalization_21/moments/Squeeze_1:output:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_21/batchnorm/add¨
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_21/batchnorm/RsqrtÚ
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:04batch_normalization_21/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_21/batchnorm/mulÎ
&batch_normalization_21/batchnorm/mul_1Muldense_17/MatMul:product:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_21/batchnorm/mul_1×
&batch_normalization_21/batchnorm/mul_2Mul/batch_normalization_21/moments/Squeeze:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_21/batchnorm/mul_2Ø
$batch_normalization_21/batchnorm/subSub2batch_normalization_21/Cast/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_21/batchnorm/subá
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_21/batchnorm/add_1v
Relu_2Relu*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¨
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_18/MatMul/ReadVariableOp
dense_18/MatMulMatMulRelu_2:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_18/MatMul¸
5batch_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_22/moments/mean/reduction_indicesç
#batch_normalization_22/moments/meanMeandense_18/MatMul:product:0>batch_normalization_22/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_22/moments/meanÁ
+batch_normalization_22/moments/StopGradientStopGradient,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_22/moments/StopGradientü
0batch_normalization_22/moments/SquaredDifferenceSquaredDifferencedense_18/MatMul:product:04batch_normalization_22/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_22/moments/SquaredDifferenceÀ
9batch_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_22/moments/variance/reduction_indices
'batch_normalization_22/moments/varianceMean4batch_normalization_22/moments/SquaredDifference:z:0Bbatch_normalization_22/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_22/moments/varianceÅ
&batch_normalization_22/moments/SqueezeSqueeze,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_22/moments/SqueezeÍ
(batch_normalization_22/moments/Squeeze_1Squeeze0batch_normalization_22/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_22/moments/Squeeze_1¡
,batch_normalization_22/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_22/AssignMovingAvg/decayÉ
+batch_normalization_22/AssignMovingAvg/CastCast5batch_normalization_22/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_22/AssignMovingAvg/Casté
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_22/AssignMovingAvg/ReadVariableOpô
*batch_normalization_22/AssignMovingAvg/subSub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_22/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_22/AssignMovingAvg/subå
*batch_normalization_22/AssignMovingAvg/mulMul.batch_normalization_22/AssignMovingAvg/sub:z:0/batch_normalization_22/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_22/AssignMovingAvg/mul²
&batch_normalization_22/AssignMovingAvgAssignSubVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_22/AssignMovingAvg¥
.batch_normalization_22/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_22/AssignMovingAvg_1/decayÏ
-batch_normalization_22/AssignMovingAvg_1/CastCast7batch_normalization_22/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_22/AssignMovingAvg_1/Castï
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_22/AssignMovingAvg_1/subSub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_22/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_22/AssignMovingAvg_1/subí
,batch_normalization_22/AssignMovingAvg_1/mulMul0batch_normalization_22/AssignMovingAvg_1/sub:z:01batch_normalization_22/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_22/AssignMovingAvg_1/mul¼
(batch_normalization_22/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource0batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_22/AssignMovingAvg_1È
*batch_normalization_22/Cast/ReadVariableOpReadVariableOp3batch_normalization_22_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_22/Cast/ReadVariableOpÎ
,batch_normalization_22/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_22_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_22/Cast_1/ReadVariableOp
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_22/batchnorm/add/yÞ
$batch_normalization_22/batchnorm/addAddV21batch_normalization_22/moments/Squeeze_1:output:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_22/batchnorm/add¨
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_22/batchnorm/RsqrtÚ
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:04batch_normalization_22/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_22/batchnorm/mulÎ
&batch_normalization_22/batchnorm/mul_1Muldense_18/MatMul:product:0(batch_normalization_22/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_22/batchnorm/mul_1×
&batch_normalization_22/batchnorm/mul_2Mul/batch_normalization_22/moments/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_22/batchnorm/mul_2Ø
$batch_normalization_22/batchnorm/subSub2batch_normalization_22/Cast/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_22/batchnorm/subá
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_22/batchnorm/add_1v
Relu_3Relu*batch_normalization_22/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¨
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_19/MatMul/ReadVariableOp
dense_19/MatMulMatMulRelu_3:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_19/MatMul§
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_19/BiasAdd/ReadVariableOp¥
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_19/BiasAddt
IdentityIdentitydense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity¿
NoOpNoOp'^batch_normalization_18/AssignMovingAvg6^batch_normalization_18/AssignMovingAvg/ReadVariableOp)^batch_normalization_18/AssignMovingAvg_18^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_18/Cast/ReadVariableOp-^batch_normalization_18/Cast_1/ReadVariableOp'^batch_normalization_19/AssignMovingAvg6^batch_normalization_19/AssignMovingAvg/ReadVariableOp)^batch_normalization_19/AssignMovingAvg_18^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_19/Cast/ReadVariableOp-^batch_normalization_19/Cast_1/ReadVariableOp'^batch_normalization_20/AssignMovingAvg6^batch_normalization_20/AssignMovingAvg/ReadVariableOp)^batch_normalization_20/AssignMovingAvg_18^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_20/Cast/ReadVariableOp-^batch_normalization_20/Cast_1/ReadVariableOp'^batch_normalization_21/AssignMovingAvg6^batch_normalization_21/AssignMovingAvg/ReadVariableOp)^batch_normalization_21/AssignMovingAvg_18^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_21/Cast/ReadVariableOp-^batch_normalization_21/Cast_1/ReadVariableOp'^batch_normalization_22/AssignMovingAvg6^batch_normalization_22/AssignMovingAvg/ReadVariableOp)^batch_normalization_22/AssignMovingAvg_18^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_22/Cast/ReadVariableOp-^batch_normalization_22/Cast_1/ReadVariableOp^dense_15/MatMul/ReadVariableOp^dense_16/MatMul/ReadVariableOp^dense_17/MatMul/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_18/AssignMovingAvg&batch_normalization_18/AssignMovingAvg2n
5batch_normalization_18/AssignMovingAvg/ReadVariableOp5batch_normalization_18/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_18/AssignMovingAvg_1(batch_normalization_18/AssignMovingAvg_12r
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_18/Cast/ReadVariableOp*batch_normalization_18/Cast/ReadVariableOp2\
,batch_normalization_18/Cast_1/ReadVariableOp,batch_normalization_18/Cast_1/ReadVariableOp2P
&batch_normalization_19/AssignMovingAvg&batch_normalization_19/AssignMovingAvg2n
5batch_normalization_19/AssignMovingAvg/ReadVariableOp5batch_normalization_19/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_19/AssignMovingAvg_1(batch_normalization_19/AssignMovingAvg_12r
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_19/Cast/ReadVariableOp*batch_normalization_19/Cast/ReadVariableOp2\
,batch_normalization_19/Cast_1/ReadVariableOp,batch_normalization_19/Cast_1/ReadVariableOp2P
&batch_normalization_20/AssignMovingAvg&batch_normalization_20/AssignMovingAvg2n
5batch_normalization_20/AssignMovingAvg/ReadVariableOp5batch_normalization_20/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_20/AssignMovingAvg_1(batch_normalization_20/AssignMovingAvg_12r
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_20/Cast/ReadVariableOp*batch_normalization_20/Cast/ReadVariableOp2\
,batch_normalization_20/Cast_1/ReadVariableOp,batch_normalization_20/Cast_1/ReadVariableOp2P
&batch_normalization_21/AssignMovingAvg&batch_normalization_21/AssignMovingAvg2n
5batch_normalization_21/AssignMovingAvg/ReadVariableOp5batch_normalization_21/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_21/AssignMovingAvg_1(batch_normalization_21/AssignMovingAvg_12r
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_21/Cast/ReadVariableOp*batch_normalization_21/Cast/ReadVariableOp2\
,batch_normalization_21/Cast_1/ReadVariableOp,batch_normalization_21/Cast_1/ReadVariableOp2P
&batch_normalization_22/AssignMovingAvg&batch_normalization_22/AssignMovingAvg2n
5batch_normalization_22/AssignMovingAvg/ReadVariableOp5batch_normalization_22/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_22/AssignMovingAvg_1(batch_normalization_22/AssignMovingAvg_12r
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_22/Cast/ReadVariableOp*batch_normalization_22/Cast/ReadVariableOp2\
,batch_normalization_22/Cast_1/ReadVariableOp,batch_normalization_22/Cast_1/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
±

R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_318781

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
×
Ò
7__inference_batch_normalization_21_layer_call_fn_318925

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
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_3169412
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
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_319063

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
)__inference_dense_18_layer_call_fn_319112

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
D__inference_dense_18_layer_call_and_return_conditional_losses_3172732
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
ÝD
â
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_317304
x+
batch_normalization_18_317194:
+
batch_normalization_18_317196:
+
batch_normalization_18_317198:
+
batch_normalization_18_317200:
!
dense_15_317211:
+
batch_normalization_19_317214:+
batch_normalization_19_317216:+
batch_normalization_19_317218:+
batch_normalization_19_317220:!
dense_16_317232:+
batch_normalization_20_317235:+
batch_normalization_20_317237:+
batch_normalization_20_317239:+
batch_normalization_20_317241:!
dense_17_317253:+
batch_normalization_21_317256:+
batch_normalization_21_317258:+
batch_normalization_21_317260:+
batch_normalization_21_317262:!
dense_18_317274:+
batch_normalization_22_317277:+
batch_normalization_22_317279:+
batch_normalization_22_317281:+
batch_normalization_22_317283:!
dense_19_317298:

dense_19_317300:

identity¢.batch_normalization_18/StatefulPartitionedCall¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢.batch_normalization_21/StatefulPartitionedCall¢.batch_normalization_22/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_18_317194batch_normalization_18_317196batch_normalization_18_317198batch_normalization_18_317200*
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
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_31638120
.batch_normalization_18/StatefulPartitionedCall²
 dense_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0dense_15_317211*
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
D__inference_dense_15_layer_call_and_return_conditional_losses_3172102"
 dense_15/StatefulPartitionedCall¿
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0batch_normalization_19_317214batch_normalization_19_317216batch_normalization_19_317218batch_normalization_19_317220*
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
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_31654720
.batch_normalization_19/StatefulPartitionedCall
ReluRelu7batch_normalization_19/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
 dense_16/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_16_317232*
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
D__inference_dense_16_layer_call_and_return_conditional_losses_3172312"
 dense_16/StatefulPartitionedCall¿
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_20_317235batch_normalization_20_317237batch_normalization_20_317239batch_normalization_20_317241*
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
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_31671320
.batch_normalization_20/StatefulPartitionedCall
Relu_1Relu7batch_normalization_20/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1
 dense_17/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_17_317253*
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
D__inference_dense_17_layer_call_and_return_conditional_losses_3172522"
 dense_17/StatefulPartitionedCall¿
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0batch_normalization_21_317256batch_normalization_21_317258batch_normalization_21_317260batch_normalization_21_317262*
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
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_31687920
.batch_normalization_21/StatefulPartitionedCall
Relu_2Relu7batch_normalization_21/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2
 dense_18/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_18_317274*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_3172732"
 dense_18/StatefulPartitionedCall¿
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0batch_normalization_22_317277batch_normalization_22_317279batch_normalization_22_317281batch_normalization_22_317283*
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
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_31704520
.batch_normalization_22/StatefulPartitionedCall
Relu_3Relu7batch_normalization_22/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¢
 dense_19/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_19_317298dense_19_317300*
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
D__inference_dense_19_layer_call_and_return_conditional_losses_3172972"
 dense_19/StatefulPartitionedCall
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityò
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
ë+
Ó
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_318981

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
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_316547

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
¦

õ
D__inference_dense_19_layer_call_and_return_conditional_losses_319138

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
7__inference_feed_forward_sub_net_3_layer_call_fn_317955
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
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_3173042
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
Ü
ú
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_318361
xL
>batch_normalization_18_assignmovingavg_readvariableop_resource:
N
@batch_normalization_18_assignmovingavg_1_readvariableop_resource:
A
3batch_normalization_18_cast_readvariableop_resource:
C
5batch_normalization_18_cast_1_readvariableop_resource:
9
'dense_15_matmul_readvariableop_resource:
L
>batch_normalization_19_assignmovingavg_readvariableop_resource:N
@batch_normalization_19_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_19_cast_readvariableop_resource:C
5batch_normalization_19_cast_1_readvariableop_resource:9
'dense_16_matmul_readvariableop_resource:L
>batch_normalization_20_assignmovingavg_readvariableop_resource:N
@batch_normalization_20_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_20_cast_readvariableop_resource:C
5batch_normalization_20_cast_1_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:L
>batch_normalization_21_assignmovingavg_readvariableop_resource:N
@batch_normalization_21_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_21_cast_readvariableop_resource:C
5batch_normalization_21_cast_1_readvariableop_resource:9
'dense_18_matmul_readvariableop_resource:L
>batch_normalization_22_assignmovingavg_readvariableop_resource:N
@batch_normalization_22_assignmovingavg_1_readvariableop_resource:A
3batch_normalization_22_cast_readvariableop_resource:C
5batch_normalization_22_cast_1_readvariableop_resource:9
'dense_19_matmul_readvariableop_resource:
6
(dense_19_biasadd_readvariableop_resource:

identity¢&batch_normalization_18/AssignMovingAvg¢5batch_normalization_18/AssignMovingAvg/ReadVariableOp¢(batch_normalization_18/AssignMovingAvg_1¢7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_18/Cast/ReadVariableOp¢,batch_normalization_18/Cast_1/ReadVariableOp¢&batch_normalization_19/AssignMovingAvg¢5batch_normalization_19/AssignMovingAvg/ReadVariableOp¢(batch_normalization_19/AssignMovingAvg_1¢7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_19/Cast/ReadVariableOp¢,batch_normalization_19/Cast_1/ReadVariableOp¢&batch_normalization_20/AssignMovingAvg¢5batch_normalization_20/AssignMovingAvg/ReadVariableOp¢(batch_normalization_20/AssignMovingAvg_1¢7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_20/Cast/ReadVariableOp¢,batch_normalization_20/Cast_1/ReadVariableOp¢&batch_normalization_21/AssignMovingAvg¢5batch_normalization_21/AssignMovingAvg/ReadVariableOp¢(batch_normalization_21/AssignMovingAvg_1¢7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_21/Cast/ReadVariableOp¢,batch_normalization_21/Cast_1/ReadVariableOp¢&batch_normalization_22/AssignMovingAvg¢5batch_normalization_22/AssignMovingAvg/ReadVariableOp¢(batch_normalization_22/AssignMovingAvg_1¢7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp¢*batch_normalization_22/Cast/ReadVariableOp¢,batch_normalization_22/Cast_1/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¸
5batch_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_18/moments/mean/reduction_indicesÏ
#batch_normalization_18/moments/meanMeanx>batch_normalization_18/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_18/moments/meanÁ
+batch_normalization_18/moments/StopGradientStopGradient,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_18/moments/StopGradientä
0batch_normalization_18/moments/SquaredDifferenceSquaredDifferencex4batch_normalization_18/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
22
0batch_normalization_18/moments/SquaredDifferenceÀ
9batch_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_18/moments/variance/reduction_indices
'batch_normalization_18/moments/varianceMean4batch_normalization_18/moments/SquaredDifference:z:0Bbatch_normalization_18/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_18/moments/varianceÅ
&batch_normalization_18/moments/SqueezeSqueeze,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_18/moments/SqueezeÍ
(batch_normalization_18/moments/Squeeze_1Squeeze0batch_normalization_18/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_18/moments/Squeeze_1¡
,batch_normalization_18/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_18/AssignMovingAvg/decayÉ
+batch_normalization_18/AssignMovingAvg/CastCast5batch_normalization_18/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_18/AssignMovingAvg/Casté
5batch_normalization_18/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource*
_output_shapes
:
*
dtype027
5batch_normalization_18/AssignMovingAvg/ReadVariableOpô
*batch_normalization_18/AssignMovingAvg/subSub=batch_normalization_18/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_18/moments/Squeeze:output:0*
T0*
_output_shapes
:
2,
*batch_normalization_18/AssignMovingAvg/subå
*batch_normalization_18/AssignMovingAvg/mulMul.batch_normalization_18/AssignMovingAvg/sub:z:0/batch_normalization_18/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:
2,
*batch_normalization_18/AssignMovingAvg/mul²
&batch_normalization_18/AssignMovingAvgAssignSubVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource.batch_normalization_18/AssignMovingAvg/mul:z:06^batch_normalization_18/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_18/AssignMovingAvg¥
.batch_normalization_18/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_18/AssignMovingAvg_1/decayÏ
-batch_normalization_18/AssignMovingAvg_1/CastCast7batch_normalization_18/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_18/AssignMovingAvg_1/Castï
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource*
_output_shapes
:
*
dtype029
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_18/AssignMovingAvg_1/subSub?batch_normalization_18/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_18/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
2.
,batch_normalization_18/AssignMovingAvg_1/subí
,batch_normalization_18/AssignMovingAvg_1/mulMul0batch_normalization_18/AssignMovingAvg_1/sub:z:01batch_normalization_18/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:
2.
,batch_normalization_18/AssignMovingAvg_1/mul¼
(batch_normalization_18/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource0batch_normalization_18/AssignMovingAvg_1/mul:z:08^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_18/AssignMovingAvg_1È
*batch_normalization_18/Cast/ReadVariableOpReadVariableOp3batch_normalization_18_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_18/Cast/ReadVariableOpÎ
,batch_normalization_18/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_18_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_18/Cast_1/ReadVariableOp
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_18/batchnorm/add/yÞ
$batch_normalization_18/batchnorm/addAddV21batch_normalization_18/moments/Squeeze_1:output:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_18/batchnorm/add¨
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_18/batchnorm/RsqrtÚ
$batch_normalization_18/batchnorm/mulMul*batch_normalization_18/batchnorm/Rsqrt:y:04batch_normalization_18/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_18/batchnorm/mul¶
&batch_normalization_18/batchnorm/mul_1Mulx(batch_normalization_18/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_18/batchnorm/mul_1×
&batch_normalization_18/batchnorm/mul_2Mul/batch_normalization_18/moments/Squeeze:output:0(batch_normalization_18/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_18/batchnorm/mul_2Ø
$batch_normalization_18/batchnorm/subSub2batch_normalization_18/Cast/ReadVariableOp:value:0*batch_normalization_18/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_18/batchnorm/subá
&batch_normalization_18/batchnorm/add_1AddV2*batch_normalization_18/batchnorm/mul_1:z:0(batch_normalization_18/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_18/batchnorm/add_1¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_15/MatMul/ReadVariableOp²
dense_15/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMul¸
5batch_normalization_19/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_19/moments/mean/reduction_indicesç
#batch_normalization_19/moments/meanMeandense_15/MatMul:product:0>batch_normalization_19/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_19/moments/meanÁ
+batch_normalization_19/moments/StopGradientStopGradient,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_19/moments/StopGradientü
0batch_normalization_19/moments/SquaredDifferenceSquaredDifferencedense_15/MatMul:product:04batch_normalization_19/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_19/moments/SquaredDifferenceÀ
9batch_normalization_19/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_19/moments/variance/reduction_indices
'batch_normalization_19/moments/varianceMean4batch_normalization_19/moments/SquaredDifference:z:0Bbatch_normalization_19/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_19/moments/varianceÅ
&batch_normalization_19/moments/SqueezeSqueeze,batch_normalization_19/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_19/moments/SqueezeÍ
(batch_normalization_19/moments/Squeeze_1Squeeze0batch_normalization_19/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_19/moments/Squeeze_1¡
,batch_normalization_19/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_19/AssignMovingAvg/decayÉ
+batch_normalization_19/AssignMovingAvg/CastCast5batch_normalization_19/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_19/AssignMovingAvg/Casté
5batch_normalization_19/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_19_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_19/AssignMovingAvg/ReadVariableOpô
*batch_normalization_19/AssignMovingAvg/subSub=batch_normalization_19/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_19/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_19/AssignMovingAvg/subå
*batch_normalization_19/AssignMovingAvg/mulMul.batch_normalization_19/AssignMovingAvg/sub:z:0/batch_normalization_19/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_19/AssignMovingAvg/mul²
&batch_normalization_19/AssignMovingAvgAssignSubVariableOp>batch_normalization_19_assignmovingavg_readvariableop_resource.batch_normalization_19/AssignMovingAvg/mul:z:06^batch_normalization_19/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_19/AssignMovingAvg¥
.batch_normalization_19/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_19/AssignMovingAvg_1/decayÏ
-batch_normalization_19/AssignMovingAvg_1/CastCast7batch_normalization_19/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_19/AssignMovingAvg_1/Castï
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_19_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_19/AssignMovingAvg_1/subSub?batch_normalization_19/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_19/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_19/AssignMovingAvg_1/subí
,batch_normalization_19/AssignMovingAvg_1/mulMul0batch_normalization_19/AssignMovingAvg_1/sub:z:01batch_normalization_19/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_19/AssignMovingAvg_1/mul¼
(batch_normalization_19/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_19_assignmovingavg_1_readvariableop_resource0batch_normalization_19/AssignMovingAvg_1/mul:z:08^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_19/AssignMovingAvg_1È
*batch_normalization_19/Cast/ReadVariableOpReadVariableOp3batch_normalization_19_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_19/Cast/ReadVariableOpÎ
,batch_normalization_19/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_19_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_19/Cast_1/ReadVariableOp
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_19/batchnorm/add/yÞ
$batch_normalization_19/batchnorm/addAddV21batch_normalization_19/moments/Squeeze_1:output:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/add¨
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_19/batchnorm/RsqrtÚ
$batch_normalization_19/batchnorm/mulMul*batch_normalization_19/batchnorm/Rsqrt:y:04batch_normalization_19/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/mulÎ
&batch_normalization_19/batchnorm/mul_1Muldense_15/MatMul:product:0(batch_normalization_19/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_19/batchnorm/mul_1×
&batch_normalization_19/batchnorm/mul_2Mul/batch_normalization_19/moments/Squeeze:output:0(batch_normalization_19/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_19/batchnorm/mul_2Ø
$batch_normalization_19/batchnorm/subSub2batch_normalization_19/Cast/ReadVariableOp:value:0*batch_normalization_19/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/subá
&batch_normalization_19/batchnorm/add_1AddV2*batch_normalization_19/batchnorm/mul_1:z:0(batch_normalization_19/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_19/batchnorm/add_1r
ReluRelu*batch_normalization_19/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¨
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_16/MatMul/ReadVariableOp
dense_16/MatMulMatMulRelu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/MatMul¸
5batch_normalization_20/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_20/moments/mean/reduction_indicesç
#batch_normalization_20/moments/meanMeandense_16/MatMul:product:0>batch_normalization_20/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_20/moments/meanÁ
+batch_normalization_20/moments/StopGradientStopGradient,batch_normalization_20/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_20/moments/StopGradientü
0batch_normalization_20/moments/SquaredDifferenceSquaredDifferencedense_16/MatMul:product:04batch_normalization_20/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_20/moments/SquaredDifferenceÀ
9batch_normalization_20/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_20/moments/variance/reduction_indices
'batch_normalization_20/moments/varianceMean4batch_normalization_20/moments/SquaredDifference:z:0Bbatch_normalization_20/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_20/moments/varianceÅ
&batch_normalization_20/moments/SqueezeSqueeze,batch_normalization_20/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_20/moments/SqueezeÍ
(batch_normalization_20/moments/Squeeze_1Squeeze0batch_normalization_20/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_20/moments/Squeeze_1¡
,batch_normalization_20/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_20/AssignMovingAvg/decayÉ
+batch_normalization_20/AssignMovingAvg/CastCast5batch_normalization_20/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_20/AssignMovingAvg/Casté
5batch_normalization_20/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_20_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_20/AssignMovingAvg/ReadVariableOpô
*batch_normalization_20/AssignMovingAvg/subSub=batch_normalization_20/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_20/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_20/AssignMovingAvg/subå
*batch_normalization_20/AssignMovingAvg/mulMul.batch_normalization_20/AssignMovingAvg/sub:z:0/batch_normalization_20/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_20/AssignMovingAvg/mul²
&batch_normalization_20/AssignMovingAvgAssignSubVariableOp>batch_normalization_20_assignmovingavg_readvariableop_resource.batch_normalization_20/AssignMovingAvg/mul:z:06^batch_normalization_20/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_20/AssignMovingAvg¥
.batch_normalization_20/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_20/AssignMovingAvg_1/decayÏ
-batch_normalization_20/AssignMovingAvg_1/CastCast7batch_normalization_20/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_20/AssignMovingAvg_1/Castï
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_20_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_20/AssignMovingAvg_1/subSub?batch_normalization_20/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_20/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_20/AssignMovingAvg_1/subí
,batch_normalization_20/AssignMovingAvg_1/mulMul0batch_normalization_20/AssignMovingAvg_1/sub:z:01batch_normalization_20/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_20/AssignMovingAvg_1/mul¼
(batch_normalization_20/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_20_assignmovingavg_1_readvariableop_resource0batch_normalization_20/AssignMovingAvg_1/mul:z:08^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_20/AssignMovingAvg_1È
*batch_normalization_20/Cast/ReadVariableOpReadVariableOp3batch_normalization_20_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_20/Cast/ReadVariableOpÎ
,batch_normalization_20/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_20_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_20/Cast_1/ReadVariableOp
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_20/batchnorm/add/yÞ
$batch_normalization_20/batchnorm/addAddV21batch_normalization_20/moments/Squeeze_1:output:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_20/batchnorm/add¨
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_20/batchnorm/RsqrtÚ
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:04batch_normalization_20/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_20/batchnorm/mulÎ
&batch_normalization_20/batchnorm/mul_1Muldense_16/MatMul:product:0(batch_normalization_20/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_20/batchnorm/mul_1×
&batch_normalization_20/batchnorm/mul_2Mul/batch_normalization_20/moments/Squeeze:output:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_20/batchnorm/mul_2Ø
$batch_normalization_20/batchnorm/subSub2batch_normalization_20/Cast/ReadVariableOp:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_20/batchnorm/subá
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_20/batchnorm/add_1v
Relu_1Relu*batch_normalization_20/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1¨
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_17/MatMul/ReadVariableOp
dense_17/MatMulMatMulRelu_1:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/MatMul¸
5batch_normalization_21/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_21/moments/mean/reduction_indicesç
#batch_normalization_21/moments/meanMeandense_17/MatMul:product:0>batch_normalization_21/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_21/moments/meanÁ
+batch_normalization_21/moments/StopGradientStopGradient,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_21/moments/StopGradientü
0batch_normalization_21/moments/SquaredDifferenceSquaredDifferencedense_17/MatMul:product:04batch_normalization_21/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_21/moments/SquaredDifferenceÀ
9batch_normalization_21/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_21/moments/variance/reduction_indices
'batch_normalization_21/moments/varianceMean4batch_normalization_21/moments/SquaredDifference:z:0Bbatch_normalization_21/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_21/moments/varianceÅ
&batch_normalization_21/moments/SqueezeSqueeze,batch_normalization_21/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_21/moments/SqueezeÍ
(batch_normalization_21/moments/Squeeze_1Squeeze0batch_normalization_21/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_21/moments/Squeeze_1¡
,batch_normalization_21/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_21/AssignMovingAvg/decayÉ
+batch_normalization_21/AssignMovingAvg/CastCast5batch_normalization_21/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_21/AssignMovingAvg/Casté
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_21/AssignMovingAvg/ReadVariableOpô
*batch_normalization_21/AssignMovingAvg/subSub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_21/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_21/AssignMovingAvg/subå
*batch_normalization_21/AssignMovingAvg/mulMul.batch_normalization_21/AssignMovingAvg/sub:z:0/batch_normalization_21/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_21/AssignMovingAvg/mul²
&batch_normalization_21/AssignMovingAvgAssignSubVariableOp>batch_normalization_21_assignmovingavg_readvariableop_resource.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_21/AssignMovingAvg¥
.batch_normalization_21/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_21/AssignMovingAvg_1/decayÏ
-batch_normalization_21/AssignMovingAvg_1/CastCast7batch_normalization_21/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_21/AssignMovingAvg_1/Castï
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_21/AssignMovingAvg_1/subSub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_21/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_21/AssignMovingAvg_1/subí
,batch_normalization_21/AssignMovingAvg_1/mulMul0batch_normalization_21/AssignMovingAvg_1/sub:z:01batch_normalization_21/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_21/AssignMovingAvg_1/mul¼
(batch_normalization_21/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_21_assignmovingavg_1_readvariableop_resource0batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_21/AssignMovingAvg_1È
*batch_normalization_21/Cast/ReadVariableOpReadVariableOp3batch_normalization_21_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_21/Cast/ReadVariableOpÎ
,batch_normalization_21/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_21_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_21/Cast_1/ReadVariableOp
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_21/batchnorm/add/yÞ
$batch_normalization_21/batchnorm/addAddV21batch_normalization_21/moments/Squeeze_1:output:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_21/batchnorm/add¨
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_21/batchnorm/RsqrtÚ
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:04batch_normalization_21/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_21/batchnorm/mulÎ
&batch_normalization_21/batchnorm/mul_1Muldense_17/MatMul:product:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_21/batchnorm/mul_1×
&batch_normalization_21/batchnorm/mul_2Mul/batch_normalization_21/moments/Squeeze:output:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_21/batchnorm/mul_2Ø
$batch_normalization_21/batchnorm/subSub2batch_normalization_21/Cast/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_21/batchnorm/subá
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_21/batchnorm/add_1v
Relu_2Relu*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¨
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_18/MatMul/ReadVariableOp
dense_18/MatMulMatMulRelu_2:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_18/MatMul¸
5batch_normalization_22/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_22/moments/mean/reduction_indicesç
#batch_normalization_22/moments/meanMeandense_18/MatMul:product:0>batch_normalization_22/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2%
#batch_normalization_22/moments/meanÁ
+batch_normalization_22/moments/StopGradientStopGradient,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes

:2-
+batch_normalization_22/moments/StopGradientü
0batch_normalization_22/moments/SquaredDifferenceSquaredDifferencedense_18/MatMul:product:04batch_normalization_22/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0batch_normalization_22/moments/SquaredDifferenceÀ
9batch_normalization_22/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_22/moments/variance/reduction_indices
'batch_normalization_22/moments/varianceMean4batch_normalization_22/moments/SquaredDifference:z:0Bbatch_normalization_22/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2)
'batch_normalization_22/moments/varianceÅ
&batch_normalization_22/moments/SqueezeSqueeze,batch_normalization_22/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2(
&batch_normalization_22/moments/SqueezeÍ
(batch_normalization_22/moments/Squeeze_1Squeeze0batch_normalization_22/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2*
(batch_normalization_22/moments/Squeeze_1¡
,batch_normalization_22/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batch_normalization_22/AssignMovingAvg/decayÉ
+batch_normalization_22/AssignMovingAvg/CastCast5batch_normalization_22/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_22/AssignMovingAvg/Casté
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_22/AssignMovingAvg/ReadVariableOpô
*batch_normalization_22/AssignMovingAvg/subSub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_22/moments/Squeeze:output:0*
T0*
_output_shapes
:2,
*batch_normalization_22/AssignMovingAvg/subå
*batch_normalization_22/AssignMovingAvg/mulMul.batch_normalization_22/AssignMovingAvg/sub:z:0/batch_normalization_22/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:2,
*batch_normalization_22/AssignMovingAvg/mul²
&batch_normalization_22/AssignMovingAvgAssignSubVariableOp>batch_normalization_22_assignmovingavg_readvariableop_resource.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_22/AssignMovingAvg¥
.batch_normalization_22/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<20
.batch_normalization_22/AssignMovingAvg_1/decayÏ
-batch_normalization_22/AssignMovingAvg_1/CastCast7batch_normalization_22/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_22/AssignMovingAvg_1/Castï
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype029
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpü
,batch_normalization_22/AssignMovingAvg_1/subSub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_22/moments/Squeeze_1:output:0*
T0*
_output_shapes
:2.
,batch_normalization_22/AssignMovingAvg_1/subí
,batch_normalization_22/AssignMovingAvg_1/mulMul0batch_normalization_22/AssignMovingAvg_1/sub:z:01batch_normalization_22/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:2.
,batch_normalization_22/AssignMovingAvg_1/mul¼
(batch_normalization_22/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_22_assignmovingavg_1_readvariableop_resource0batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_22/AssignMovingAvg_1È
*batch_normalization_22/Cast/ReadVariableOpReadVariableOp3batch_normalization_22_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_22/Cast/ReadVariableOpÎ
,batch_normalization_22/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_22_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_22/Cast_1/ReadVariableOp
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_22/batchnorm/add/yÞ
$batch_normalization_22/batchnorm/addAddV21batch_normalization_22/moments/Squeeze_1:output:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_22/batchnorm/add¨
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_22/batchnorm/RsqrtÚ
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:04batch_normalization_22/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_22/batchnorm/mulÎ
&batch_normalization_22/batchnorm/mul_1Muldense_18/MatMul:product:0(batch_normalization_22/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_22/batchnorm/mul_1×
&batch_normalization_22/batchnorm/mul_2Mul/batch_normalization_22/moments/Squeeze:output:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_22/batchnorm/mul_2Ø
$batch_normalization_22/batchnorm/subSub2batch_normalization_22/Cast/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_22/batchnorm/subá
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_22/batchnorm/add_1v
Relu_3Relu*batch_normalization_22/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¨
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_19/MatMul/ReadVariableOp
dense_19/MatMulMatMulRelu_3:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_19/MatMul§
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_19/BiasAdd/ReadVariableOp¥
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_19/BiasAddt
IdentityIdentitydense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity¿
NoOpNoOp'^batch_normalization_18/AssignMovingAvg6^batch_normalization_18/AssignMovingAvg/ReadVariableOp)^batch_normalization_18/AssignMovingAvg_18^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_18/Cast/ReadVariableOp-^batch_normalization_18/Cast_1/ReadVariableOp'^batch_normalization_19/AssignMovingAvg6^batch_normalization_19/AssignMovingAvg/ReadVariableOp)^batch_normalization_19/AssignMovingAvg_18^batch_normalization_19/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_19/Cast/ReadVariableOp-^batch_normalization_19/Cast_1/ReadVariableOp'^batch_normalization_20/AssignMovingAvg6^batch_normalization_20/AssignMovingAvg/ReadVariableOp)^batch_normalization_20/AssignMovingAvg_18^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_20/Cast/ReadVariableOp-^batch_normalization_20/Cast_1/ReadVariableOp'^batch_normalization_21/AssignMovingAvg6^batch_normalization_21/AssignMovingAvg/ReadVariableOp)^batch_normalization_21/AssignMovingAvg_18^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_21/Cast/ReadVariableOp-^batch_normalization_21/Cast_1/ReadVariableOp'^batch_normalization_22/AssignMovingAvg6^batch_normalization_22/AssignMovingAvg/ReadVariableOp)^batch_normalization_22/AssignMovingAvg_18^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_22/Cast/ReadVariableOp-^batch_normalization_22/Cast_1/ReadVariableOp^dense_15/MatMul/ReadVariableOp^dense_16/MatMul/ReadVariableOp^dense_17/MatMul/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_18/AssignMovingAvg&batch_normalization_18/AssignMovingAvg2n
5batch_normalization_18/AssignMovingAvg/ReadVariableOp5batch_normalization_18/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_18/AssignMovingAvg_1(batch_normalization_18/AssignMovingAvg_12r
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_18/Cast/ReadVariableOp*batch_normalization_18/Cast/ReadVariableOp2\
,batch_normalization_18/Cast_1/ReadVariableOp,batch_normalization_18/Cast_1/ReadVariableOp2P
&batch_normalization_19/AssignMovingAvg&batch_normalization_19/AssignMovingAvg2n
5batch_normalization_19/AssignMovingAvg/ReadVariableOp5batch_normalization_19/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_19/AssignMovingAvg_1(batch_normalization_19/AssignMovingAvg_12r
7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp7batch_normalization_19/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_19/Cast/ReadVariableOp*batch_normalization_19/Cast/ReadVariableOp2\
,batch_normalization_19/Cast_1/ReadVariableOp,batch_normalization_19/Cast_1/ReadVariableOp2P
&batch_normalization_20/AssignMovingAvg&batch_normalization_20/AssignMovingAvg2n
5batch_normalization_20/AssignMovingAvg/ReadVariableOp5batch_normalization_20/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_20/AssignMovingAvg_1(batch_normalization_20/AssignMovingAvg_12r
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_20/Cast/ReadVariableOp*batch_normalization_20/Cast/ReadVariableOp2\
,batch_normalization_20/Cast_1/ReadVariableOp,batch_normalization_20/Cast_1/ReadVariableOp2P
&batch_normalization_21/AssignMovingAvg&batch_normalization_21/AssignMovingAvg2n
5batch_normalization_21/AssignMovingAvg/ReadVariableOp5batch_normalization_21/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_21/AssignMovingAvg_1(batch_normalization_21/AssignMovingAvg_12r
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_21/Cast/ReadVariableOp*batch_normalization_21/Cast/ReadVariableOp2\
,batch_normalization_21/Cast_1/ReadVariableOp,batch_normalization_21/Cast_1/ReadVariableOp2P
&batch_normalization_22/AssignMovingAvg&batch_normalization_22/AssignMovingAvg2n
5batch_normalization_22/AssignMovingAvg/ReadVariableOp5batch_normalization_22/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_22/AssignMovingAvg_1(batch_normalization_22/AssignMovingAvg_12r
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_22/Cast/ReadVariableOp*batch_normalization_22/Cast/ReadVariableOp2\
,batch_normalization_22/Cast_1/ReadVariableOp,batch_normalization_22/Cast_1/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
Ù
Ò
7__inference_batch_normalization_22_layer_call_fn_318994

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
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_3170452
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
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_318817

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
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_316381

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
©©

R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_318175
xA
3batch_normalization_18_cast_readvariableop_resource:
C
5batch_normalization_18_cast_1_readvariableop_resource:
C
5batch_normalization_18_cast_2_readvariableop_resource:
C
5batch_normalization_18_cast_3_readvariableop_resource:
9
'dense_15_matmul_readvariableop_resource:
A
3batch_normalization_19_cast_readvariableop_resource:C
5batch_normalization_19_cast_1_readvariableop_resource:C
5batch_normalization_19_cast_2_readvariableop_resource:C
5batch_normalization_19_cast_3_readvariableop_resource:9
'dense_16_matmul_readvariableop_resource:A
3batch_normalization_20_cast_readvariableop_resource:C
5batch_normalization_20_cast_1_readvariableop_resource:C
5batch_normalization_20_cast_2_readvariableop_resource:C
5batch_normalization_20_cast_3_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:A
3batch_normalization_21_cast_readvariableop_resource:C
5batch_normalization_21_cast_1_readvariableop_resource:C
5batch_normalization_21_cast_2_readvariableop_resource:C
5batch_normalization_21_cast_3_readvariableop_resource:9
'dense_18_matmul_readvariableop_resource:A
3batch_normalization_22_cast_readvariableop_resource:C
5batch_normalization_22_cast_1_readvariableop_resource:C
5batch_normalization_22_cast_2_readvariableop_resource:C
5batch_normalization_22_cast_3_readvariableop_resource:9
'dense_19_matmul_readvariableop_resource:
6
(dense_19_biasadd_readvariableop_resource:

identity¢*batch_normalization_18/Cast/ReadVariableOp¢,batch_normalization_18/Cast_1/ReadVariableOp¢,batch_normalization_18/Cast_2/ReadVariableOp¢,batch_normalization_18/Cast_3/ReadVariableOp¢*batch_normalization_19/Cast/ReadVariableOp¢,batch_normalization_19/Cast_1/ReadVariableOp¢,batch_normalization_19/Cast_2/ReadVariableOp¢,batch_normalization_19/Cast_3/ReadVariableOp¢*batch_normalization_20/Cast/ReadVariableOp¢,batch_normalization_20/Cast_1/ReadVariableOp¢,batch_normalization_20/Cast_2/ReadVariableOp¢,batch_normalization_20/Cast_3/ReadVariableOp¢*batch_normalization_21/Cast/ReadVariableOp¢,batch_normalization_21/Cast_1/ReadVariableOp¢,batch_normalization_21/Cast_2/ReadVariableOp¢,batch_normalization_21/Cast_3/ReadVariableOp¢*batch_normalization_22/Cast/ReadVariableOp¢,batch_normalization_22/Cast_1/ReadVariableOp¢,batch_normalization_22/Cast_2/ReadVariableOp¢,batch_normalization_22/Cast_3/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOpÈ
*batch_normalization_18/Cast/ReadVariableOpReadVariableOp3batch_normalization_18_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_18/Cast/ReadVariableOpÎ
,batch_normalization_18/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_18_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_18/Cast_1/ReadVariableOpÎ
,batch_normalization_18/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_18_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_18/Cast_2/ReadVariableOpÎ
,batch_normalization_18/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_18_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_18/Cast_3/ReadVariableOp
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_18/batchnorm/add/yá
$batch_normalization_18/batchnorm/addAddV24batch_normalization_18/Cast_1/ReadVariableOp:value:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_18/batchnorm/add¨
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_18/batchnorm/RsqrtÚ
$batch_normalization_18/batchnorm/mulMul*batch_normalization_18/batchnorm/Rsqrt:y:04batch_normalization_18/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_18/batchnorm/mul¶
&batch_normalization_18/batchnorm/mul_1Mulx(batch_normalization_18/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_18/batchnorm/mul_1Ú
&batch_normalization_18/batchnorm/mul_2Mul2batch_normalization_18/Cast/ReadVariableOp:value:0(batch_normalization_18/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_18/batchnorm/mul_2Ú
$batch_normalization_18/batchnorm/subSub4batch_normalization_18/Cast_2/ReadVariableOp:value:0*batch_normalization_18/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_18/batchnorm/subá
&batch_normalization_18/batchnorm/add_1AddV2*batch_normalization_18/batchnorm/mul_1:z:0(batch_normalization_18/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_18/batchnorm/add_1¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_15/MatMul/ReadVariableOp²
dense_15/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMulÈ
*batch_normalization_19/Cast/ReadVariableOpReadVariableOp3batch_normalization_19_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_19/Cast/ReadVariableOpÎ
,batch_normalization_19/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_19_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_19/Cast_1/ReadVariableOpÎ
,batch_normalization_19/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_19_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_19/Cast_2/ReadVariableOpÎ
,batch_normalization_19/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_19_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_19/Cast_3/ReadVariableOp
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_19/batchnorm/add/yá
$batch_normalization_19/batchnorm/addAddV24batch_normalization_19/Cast_1/ReadVariableOp:value:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/add¨
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_19/batchnorm/RsqrtÚ
$batch_normalization_19/batchnorm/mulMul*batch_normalization_19/batchnorm/Rsqrt:y:04batch_normalization_19/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/mulÎ
&batch_normalization_19/batchnorm/mul_1Muldense_15/MatMul:product:0(batch_normalization_19/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_19/batchnorm/mul_1Ú
&batch_normalization_19/batchnorm/mul_2Mul2batch_normalization_19/Cast/ReadVariableOp:value:0(batch_normalization_19/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_19/batchnorm/mul_2Ú
$batch_normalization_19/batchnorm/subSub4batch_normalization_19/Cast_2/ReadVariableOp:value:0*batch_normalization_19/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/subá
&batch_normalization_19/batchnorm/add_1AddV2*batch_normalization_19/batchnorm/mul_1:z:0(batch_normalization_19/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_19/batchnorm/add_1r
ReluRelu*batch_normalization_19/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¨
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_16/MatMul/ReadVariableOp
dense_16/MatMulMatMulRelu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/MatMulÈ
*batch_normalization_20/Cast/ReadVariableOpReadVariableOp3batch_normalization_20_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_20/Cast/ReadVariableOpÎ
,batch_normalization_20/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_20_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_20/Cast_1/ReadVariableOpÎ
,batch_normalization_20/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_20_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_20/Cast_2/ReadVariableOpÎ
,batch_normalization_20/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_20_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_20/Cast_3/ReadVariableOp
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_20/batchnorm/add/yá
$batch_normalization_20/batchnorm/addAddV24batch_normalization_20/Cast_1/ReadVariableOp:value:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_20/batchnorm/add¨
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_20/batchnorm/RsqrtÚ
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:04batch_normalization_20/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_20/batchnorm/mulÎ
&batch_normalization_20/batchnorm/mul_1Muldense_16/MatMul:product:0(batch_normalization_20/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_20/batchnorm/mul_1Ú
&batch_normalization_20/batchnorm/mul_2Mul2batch_normalization_20/Cast/ReadVariableOp:value:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_20/batchnorm/mul_2Ú
$batch_normalization_20/batchnorm/subSub4batch_normalization_20/Cast_2/ReadVariableOp:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_20/batchnorm/subá
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_20/batchnorm/add_1v
Relu_1Relu*batch_normalization_20/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1¨
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_17/MatMul/ReadVariableOp
dense_17/MatMulMatMulRelu_1:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/MatMulÈ
*batch_normalization_21/Cast/ReadVariableOpReadVariableOp3batch_normalization_21_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_21/Cast/ReadVariableOpÎ
,batch_normalization_21/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_21_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_21/Cast_1/ReadVariableOpÎ
,batch_normalization_21/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_21_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_21/Cast_2/ReadVariableOpÎ
,batch_normalization_21/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_21_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_21/Cast_3/ReadVariableOp
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_21/batchnorm/add/yá
$batch_normalization_21/batchnorm/addAddV24batch_normalization_21/Cast_1/ReadVariableOp:value:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_21/batchnorm/add¨
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_21/batchnorm/RsqrtÚ
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:04batch_normalization_21/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_21/batchnorm/mulÎ
&batch_normalization_21/batchnorm/mul_1Muldense_17/MatMul:product:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_21/batchnorm/mul_1Ú
&batch_normalization_21/batchnorm/mul_2Mul2batch_normalization_21/Cast/ReadVariableOp:value:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_21/batchnorm/mul_2Ú
$batch_normalization_21/batchnorm/subSub4batch_normalization_21/Cast_2/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_21/batchnorm/subá
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_21/batchnorm/add_1v
Relu_2Relu*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¨
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_18/MatMul/ReadVariableOp
dense_18/MatMulMatMulRelu_2:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_18/MatMulÈ
*batch_normalization_22/Cast/ReadVariableOpReadVariableOp3batch_normalization_22_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_22/Cast/ReadVariableOpÎ
,batch_normalization_22/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_22_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_22/Cast_1/ReadVariableOpÎ
,batch_normalization_22/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_22_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_22/Cast_2/ReadVariableOpÎ
,batch_normalization_22/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_22_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_22/Cast_3/ReadVariableOp
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_22/batchnorm/add/yá
$batch_normalization_22/batchnorm/addAddV24batch_normalization_22/Cast_1/ReadVariableOp:value:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_22/batchnorm/add¨
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_22/batchnorm/RsqrtÚ
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:04batch_normalization_22/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_22/batchnorm/mulÎ
&batch_normalization_22/batchnorm/mul_1Muldense_18/MatMul:product:0(batch_normalization_22/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_22/batchnorm/mul_1Ú
&batch_normalization_22/batchnorm/mul_2Mul2batch_normalization_22/Cast/ReadVariableOp:value:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_22/batchnorm/mul_2Ú
$batch_normalization_22/batchnorm/subSub4batch_normalization_22/Cast_2/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_22/batchnorm/subá
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_22/batchnorm/add_1v
Relu_3Relu*batch_normalization_22/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¨
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_19/MatMul/ReadVariableOp
dense_19/MatMulMatMulRelu_3:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_19/MatMul§
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_19/BiasAdd/ReadVariableOp¥
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_19/BiasAddt
IdentityIdentitydense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity·	
NoOpNoOp+^batch_normalization_18/Cast/ReadVariableOp-^batch_normalization_18/Cast_1/ReadVariableOp-^batch_normalization_18/Cast_2/ReadVariableOp-^batch_normalization_18/Cast_3/ReadVariableOp+^batch_normalization_19/Cast/ReadVariableOp-^batch_normalization_19/Cast_1/ReadVariableOp-^batch_normalization_19/Cast_2/ReadVariableOp-^batch_normalization_19/Cast_3/ReadVariableOp+^batch_normalization_20/Cast/ReadVariableOp-^batch_normalization_20/Cast_1/ReadVariableOp-^batch_normalization_20/Cast_2/ReadVariableOp-^batch_normalization_20/Cast_3/ReadVariableOp+^batch_normalization_21/Cast/ReadVariableOp-^batch_normalization_21/Cast_1/ReadVariableOp-^batch_normalization_21/Cast_2/ReadVariableOp-^batch_normalization_21/Cast_3/ReadVariableOp+^batch_normalization_22/Cast/ReadVariableOp-^batch_normalization_22/Cast_1/ReadVariableOp-^batch_normalization_22/Cast_2/ReadVariableOp-^batch_normalization_22/Cast_3/ReadVariableOp^dense_15/MatMul/ReadVariableOp^dense_16/MatMul/ReadVariableOp^dense_17/MatMul/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_18/Cast/ReadVariableOp*batch_normalization_18/Cast/ReadVariableOp2\
,batch_normalization_18/Cast_1/ReadVariableOp,batch_normalization_18/Cast_1/ReadVariableOp2\
,batch_normalization_18/Cast_2/ReadVariableOp,batch_normalization_18/Cast_2/ReadVariableOp2\
,batch_normalization_18/Cast_3/ReadVariableOp,batch_normalization_18/Cast_3/ReadVariableOp2X
*batch_normalization_19/Cast/ReadVariableOp*batch_normalization_19/Cast/ReadVariableOp2\
,batch_normalization_19/Cast_1/ReadVariableOp,batch_normalization_19/Cast_1/ReadVariableOp2\
,batch_normalization_19/Cast_2/ReadVariableOp,batch_normalization_19/Cast_2/ReadVariableOp2\
,batch_normalization_19/Cast_3/ReadVariableOp,batch_normalization_19/Cast_3/ReadVariableOp2X
*batch_normalization_20/Cast/ReadVariableOp*batch_normalization_20/Cast/ReadVariableOp2\
,batch_normalization_20/Cast_1/ReadVariableOp,batch_normalization_20/Cast_1/ReadVariableOp2\
,batch_normalization_20/Cast_2/ReadVariableOp,batch_normalization_20/Cast_2/ReadVariableOp2\
,batch_normalization_20/Cast_3/ReadVariableOp,batch_normalization_20/Cast_3/ReadVariableOp2X
*batch_normalization_21/Cast/ReadVariableOp*batch_normalization_21/Cast/ReadVariableOp2\
,batch_normalization_21/Cast_1/ReadVariableOp,batch_normalization_21/Cast_1/ReadVariableOp2\
,batch_normalization_21/Cast_2/ReadVariableOp,batch_normalization_21/Cast_2/ReadVariableOp2\
,batch_normalization_21/Cast_3/ReadVariableOp,batch_normalization_21/Cast_3/ReadVariableOp2X
*batch_normalization_22/Cast/ReadVariableOp*batch_normalization_22/Cast/ReadVariableOp2\
,batch_normalization_22/Cast_1/ReadVariableOp,batch_normalization_22/Cast_1/ReadVariableOp2\
,batch_normalization_22/Cast_2/ReadVariableOp,batch_normalization_22/Cast_2/ReadVariableOp2\
,batch_normalization_22/Cast_3/ReadVariableOp,batch_normalization_22/Cast_3/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
þ

7__inference_feed_forward_sub_net_3_layer_call_fn_317898
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
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_3173042
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
ë+
Ó
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_316443

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
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_316775

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
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_316609

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
)__inference_dense_15_layer_call_fn_319070

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
D__inference_dense_15_layer_call_and_return_conditional_losses_3172102
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
×
Ò
7__inference_batch_normalization_19_layer_call_fn_318761

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
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_3166092
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
D__inference_dense_19_layer_call_and_return_conditional_losses_317297

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
ÓD
â
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_317530
x+
batch_normalization_18_317463:
+
batch_normalization_18_317465:
+
batch_normalization_18_317467:
+
batch_normalization_18_317469:
!
dense_15_317472:
+
batch_normalization_19_317475:+
batch_normalization_19_317477:+
batch_normalization_19_317479:+
batch_normalization_19_317481:!
dense_16_317485:+
batch_normalization_20_317488:+
batch_normalization_20_317490:+
batch_normalization_20_317492:+
batch_normalization_20_317494:!
dense_17_317498:+
batch_normalization_21_317501:+
batch_normalization_21_317503:+
batch_normalization_21_317505:+
batch_normalization_21_317507:!
dense_18_317511:+
batch_normalization_22_317514:+
batch_normalization_22_317516:+
batch_normalization_22_317518:+
batch_normalization_22_317520:!
dense_19_317524:

dense_19_317526:

identity¢.batch_normalization_18/StatefulPartitionedCall¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢.batch_normalization_21/StatefulPartitionedCall¢.batch_normalization_22/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_18_317463batch_normalization_18_317465batch_normalization_18_317467batch_normalization_18_317469*
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
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_31644320
.batch_normalization_18/StatefulPartitionedCall²
 dense_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0dense_15_317472*
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
D__inference_dense_15_layer_call_and_return_conditional_losses_3172102"
 dense_15/StatefulPartitionedCall½
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0batch_normalization_19_317475batch_normalization_19_317477batch_normalization_19_317479batch_normalization_19_317481*
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
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_31660920
.batch_normalization_19/StatefulPartitionedCall
ReluRelu7batch_normalization_19/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
 dense_16/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_16_317485*
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
D__inference_dense_16_layer_call_and_return_conditional_losses_3172312"
 dense_16/StatefulPartitionedCall½
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0batch_normalization_20_317488batch_normalization_20_317490batch_normalization_20_317492batch_normalization_20_317494*
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
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_31677520
.batch_normalization_20/StatefulPartitionedCall
Relu_1Relu7batch_normalization_20/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1
 dense_17/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_17_317498*
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
D__inference_dense_17_layer_call_and_return_conditional_losses_3172522"
 dense_17/StatefulPartitionedCall½
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0batch_normalization_21_317501batch_normalization_21_317503batch_normalization_21_317505batch_normalization_21_317507*
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
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_31694120
.batch_normalization_21/StatefulPartitionedCall
Relu_2Relu7batch_normalization_21/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2
 dense_18/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_18_317511*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_3172732"
 dense_18/StatefulPartitionedCall½
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0batch_normalization_22_317514batch_normalization_22_317516batch_normalization_22_317518batch_normalization_22_317520*
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
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_31710720
.batch_normalization_22/StatefulPartitionedCall
Relu_3Relu7batch_normalization_22/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¢
 dense_19/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_19_317524dense_19_317526*
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
D__inference_dense_19_layer_call_and_return_conditional_losses_3172972"
 dense_19/StatefulPartitionedCall
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityò
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namex
þ
­
D__inference_dense_15_layer_call_and_return_conditional_losses_319077

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
×
Ò
7__inference_batch_normalization_22_layer_call_fn_319007

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
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_3171072
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
ô

7__inference_feed_forward_sub_net_3_layer_call_fn_318069
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
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_3175302
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
7__inference_batch_normalization_20_layer_call_fn_318843

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
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_3167752
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
»©

R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_318467
input_1A
3batch_normalization_18_cast_readvariableop_resource:
C
5batch_normalization_18_cast_1_readvariableop_resource:
C
5batch_normalization_18_cast_2_readvariableop_resource:
C
5batch_normalization_18_cast_3_readvariableop_resource:
9
'dense_15_matmul_readvariableop_resource:
A
3batch_normalization_19_cast_readvariableop_resource:C
5batch_normalization_19_cast_1_readvariableop_resource:C
5batch_normalization_19_cast_2_readvariableop_resource:C
5batch_normalization_19_cast_3_readvariableop_resource:9
'dense_16_matmul_readvariableop_resource:A
3batch_normalization_20_cast_readvariableop_resource:C
5batch_normalization_20_cast_1_readvariableop_resource:C
5batch_normalization_20_cast_2_readvariableop_resource:C
5batch_normalization_20_cast_3_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:A
3batch_normalization_21_cast_readvariableop_resource:C
5batch_normalization_21_cast_1_readvariableop_resource:C
5batch_normalization_21_cast_2_readvariableop_resource:C
5batch_normalization_21_cast_3_readvariableop_resource:9
'dense_18_matmul_readvariableop_resource:A
3batch_normalization_22_cast_readvariableop_resource:C
5batch_normalization_22_cast_1_readvariableop_resource:C
5batch_normalization_22_cast_2_readvariableop_resource:C
5batch_normalization_22_cast_3_readvariableop_resource:9
'dense_19_matmul_readvariableop_resource:
6
(dense_19_biasadd_readvariableop_resource:

identity¢*batch_normalization_18/Cast/ReadVariableOp¢,batch_normalization_18/Cast_1/ReadVariableOp¢,batch_normalization_18/Cast_2/ReadVariableOp¢,batch_normalization_18/Cast_3/ReadVariableOp¢*batch_normalization_19/Cast/ReadVariableOp¢,batch_normalization_19/Cast_1/ReadVariableOp¢,batch_normalization_19/Cast_2/ReadVariableOp¢,batch_normalization_19/Cast_3/ReadVariableOp¢*batch_normalization_20/Cast/ReadVariableOp¢,batch_normalization_20/Cast_1/ReadVariableOp¢,batch_normalization_20/Cast_2/ReadVariableOp¢,batch_normalization_20/Cast_3/ReadVariableOp¢*batch_normalization_21/Cast/ReadVariableOp¢,batch_normalization_21/Cast_1/ReadVariableOp¢,batch_normalization_21/Cast_2/ReadVariableOp¢,batch_normalization_21/Cast_3/ReadVariableOp¢*batch_normalization_22/Cast/ReadVariableOp¢,batch_normalization_22/Cast_1/ReadVariableOp¢,batch_normalization_22/Cast_2/ReadVariableOp¢,batch_normalization_22/Cast_3/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOpÈ
*batch_normalization_18/Cast/ReadVariableOpReadVariableOp3batch_normalization_18_cast_readvariableop_resource*
_output_shapes
:
*
dtype02,
*batch_normalization_18/Cast/ReadVariableOpÎ
,batch_normalization_18/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_18_cast_1_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_18/Cast_1/ReadVariableOpÎ
,batch_normalization_18/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_18_cast_2_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_18/Cast_2/ReadVariableOpÎ
,batch_normalization_18/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_18_cast_3_readvariableop_resource*
_output_shapes
:
*
dtype02.
,batch_normalization_18/Cast_3/ReadVariableOp
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_18/batchnorm/add/yá
$batch_normalization_18/batchnorm/addAddV24batch_normalization_18/Cast_1/ReadVariableOp:value:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_18/batchnorm/add¨
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_18/batchnorm/RsqrtÚ
$batch_normalization_18/batchnorm/mulMul*batch_normalization_18/batchnorm/Rsqrt:y:04batch_normalization_18/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_18/batchnorm/mul¼
&batch_normalization_18/batchnorm/mul_1Mulinput_1(batch_normalization_18/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_18/batchnorm/mul_1Ú
&batch_normalization_18/batchnorm/mul_2Mul2batch_normalization_18/Cast/ReadVariableOp:value:0(batch_normalization_18/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_18/batchnorm/mul_2Ú
$batch_normalization_18/batchnorm/subSub4batch_normalization_18/Cast_2/ReadVariableOp:value:0*batch_normalization_18/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_18/batchnorm/subá
&batch_normalization_18/batchnorm/add_1AddV2*batch_normalization_18/batchnorm/mul_1:z:0(batch_normalization_18/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&batch_normalization_18/batchnorm/add_1¨
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_15/MatMul/ReadVariableOp²
dense_15/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMulÈ
*batch_normalization_19/Cast/ReadVariableOpReadVariableOp3batch_normalization_19_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_19/Cast/ReadVariableOpÎ
,batch_normalization_19/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_19_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_19/Cast_1/ReadVariableOpÎ
,batch_normalization_19/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_19_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_19/Cast_2/ReadVariableOpÎ
,batch_normalization_19/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_19_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_19/Cast_3/ReadVariableOp
&batch_normalization_19/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_19/batchnorm/add/yá
$batch_normalization_19/batchnorm/addAddV24batch_normalization_19/Cast_1/ReadVariableOp:value:0/batch_normalization_19/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/add¨
&batch_normalization_19/batchnorm/RsqrtRsqrt(batch_normalization_19/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_19/batchnorm/RsqrtÚ
$batch_normalization_19/batchnorm/mulMul*batch_normalization_19/batchnorm/Rsqrt:y:04batch_normalization_19/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/mulÎ
&batch_normalization_19/batchnorm/mul_1Muldense_15/MatMul:product:0(batch_normalization_19/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_19/batchnorm/mul_1Ú
&batch_normalization_19/batchnorm/mul_2Mul2batch_normalization_19/Cast/ReadVariableOp:value:0(batch_normalization_19/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_19/batchnorm/mul_2Ú
$batch_normalization_19/batchnorm/subSub4batch_normalization_19/Cast_2/ReadVariableOp:value:0*batch_normalization_19/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_19/batchnorm/subá
&batch_normalization_19/batchnorm/add_1AddV2*batch_normalization_19/batchnorm/mul_1:z:0(batch_normalization_19/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_19/batchnorm/add_1r
ReluRelu*batch_normalization_19/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu¨
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_16/MatMul/ReadVariableOp
dense_16/MatMulMatMulRelu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/MatMulÈ
*batch_normalization_20/Cast/ReadVariableOpReadVariableOp3batch_normalization_20_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_20/Cast/ReadVariableOpÎ
,batch_normalization_20/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_20_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_20/Cast_1/ReadVariableOpÎ
,batch_normalization_20/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_20_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_20/Cast_2/ReadVariableOpÎ
,batch_normalization_20/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_20_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_20/Cast_3/ReadVariableOp
&batch_normalization_20/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_20/batchnorm/add/yá
$batch_normalization_20/batchnorm/addAddV24batch_normalization_20/Cast_1/ReadVariableOp:value:0/batch_normalization_20/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_20/batchnorm/add¨
&batch_normalization_20/batchnorm/RsqrtRsqrt(batch_normalization_20/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_20/batchnorm/RsqrtÚ
$batch_normalization_20/batchnorm/mulMul*batch_normalization_20/batchnorm/Rsqrt:y:04batch_normalization_20/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_20/batchnorm/mulÎ
&batch_normalization_20/batchnorm/mul_1Muldense_16/MatMul:product:0(batch_normalization_20/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_20/batchnorm/mul_1Ú
&batch_normalization_20/batchnorm/mul_2Mul2batch_normalization_20/Cast/ReadVariableOp:value:0(batch_normalization_20/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_20/batchnorm/mul_2Ú
$batch_normalization_20/batchnorm/subSub4batch_normalization_20/Cast_2/ReadVariableOp:value:0*batch_normalization_20/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_20/batchnorm/subá
&batch_normalization_20/batchnorm/add_1AddV2*batch_normalization_20/batchnorm/mul_1:z:0(batch_normalization_20/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_20/batchnorm/add_1v
Relu_1Relu*batch_normalization_20/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1¨
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_17/MatMul/ReadVariableOp
dense_17/MatMulMatMulRelu_1:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_17/MatMulÈ
*batch_normalization_21/Cast/ReadVariableOpReadVariableOp3batch_normalization_21_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_21/Cast/ReadVariableOpÎ
,batch_normalization_21/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_21_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_21/Cast_1/ReadVariableOpÎ
,batch_normalization_21/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_21_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_21/Cast_2/ReadVariableOpÎ
,batch_normalization_21/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_21_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_21/Cast_3/ReadVariableOp
&batch_normalization_21/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_21/batchnorm/add/yá
$batch_normalization_21/batchnorm/addAddV24batch_normalization_21/Cast_1/ReadVariableOp:value:0/batch_normalization_21/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_21/batchnorm/add¨
&batch_normalization_21/batchnorm/RsqrtRsqrt(batch_normalization_21/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_21/batchnorm/RsqrtÚ
$batch_normalization_21/batchnorm/mulMul*batch_normalization_21/batchnorm/Rsqrt:y:04batch_normalization_21/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_21/batchnorm/mulÎ
&batch_normalization_21/batchnorm/mul_1Muldense_17/MatMul:product:0(batch_normalization_21/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_21/batchnorm/mul_1Ú
&batch_normalization_21/batchnorm/mul_2Mul2batch_normalization_21/Cast/ReadVariableOp:value:0(batch_normalization_21/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_21/batchnorm/mul_2Ú
$batch_normalization_21/batchnorm/subSub4batch_normalization_21/Cast_2/ReadVariableOp:value:0*batch_normalization_21/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_21/batchnorm/subá
&batch_normalization_21/batchnorm/add_1AddV2*batch_normalization_21/batchnorm/mul_1:z:0(batch_normalization_21/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_21/batchnorm/add_1v
Relu_2Relu*batch_normalization_21/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_2¨
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_18/MatMul/ReadVariableOp
dense_18/MatMulMatMulRelu_2:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_18/MatMulÈ
*batch_normalization_22/Cast/ReadVariableOpReadVariableOp3batch_normalization_22_cast_readvariableop_resource*
_output_shapes
:*
dtype02,
*batch_normalization_22/Cast/ReadVariableOpÎ
,batch_normalization_22/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_22_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_22/Cast_1/ReadVariableOpÎ
,batch_normalization_22/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_22_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_22/Cast_2/ReadVariableOpÎ
,batch_normalization_22/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_22_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization_22/Cast_3/ReadVariableOp
&batch_normalization_22/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2íµ ÷Æ°>2(
&batch_normalization_22/batchnorm/add/yá
$batch_normalization_22/batchnorm/addAddV24batch_normalization_22/Cast_1/ReadVariableOp:value:0/batch_normalization_22/batchnorm/add/y:output:0*
T0*
_output_shapes
:2&
$batch_normalization_22/batchnorm/add¨
&batch_normalization_22/batchnorm/RsqrtRsqrt(batch_normalization_22/batchnorm/add:z:0*
T0*
_output_shapes
:2(
&batch_normalization_22/batchnorm/RsqrtÚ
$batch_normalization_22/batchnorm/mulMul*batch_normalization_22/batchnorm/Rsqrt:y:04batch_normalization_22/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2&
$batch_normalization_22/batchnorm/mulÎ
&batch_normalization_22/batchnorm/mul_1Muldense_18/MatMul:product:0(batch_normalization_22/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_22/batchnorm/mul_1Ú
&batch_normalization_22/batchnorm/mul_2Mul2batch_normalization_22/Cast/ReadVariableOp:value:0(batch_normalization_22/batchnorm/mul:z:0*
T0*
_output_shapes
:2(
&batch_normalization_22/batchnorm/mul_2Ú
$batch_normalization_22/batchnorm/subSub4batch_normalization_22/Cast_2/ReadVariableOp:value:0*batch_normalization_22/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2&
$batch_normalization_22/batchnorm/subá
&batch_normalization_22/batchnorm/add_1AddV2*batch_normalization_22/batchnorm/mul_1:z:0(batch_normalization_22/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&batch_normalization_22/batchnorm/add_1v
Relu_3Relu*batch_normalization_22/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_3¨
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_19/MatMul/ReadVariableOp
dense_19/MatMulMatMulRelu_3:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_19/MatMul§
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_19/BiasAdd/ReadVariableOp¥
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_19/BiasAddt
IdentityIdentitydense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity·	
NoOpNoOp+^batch_normalization_18/Cast/ReadVariableOp-^batch_normalization_18/Cast_1/ReadVariableOp-^batch_normalization_18/Cast_2/ReadVariableOp-^batch_normalization_18/Cast_3/ReadVariableOp+^batch_normalization_19/Cast/ReadVariableOp-^batch_normalization_19/Cast_1/ReadVariableOp-^batch_normalization_19/Cast_2/ReadVariableOp-^batch_normalization_19/Cast_3/ReadVariableOp+^batch_normalization_20/Cast/ReadVariableOp-^batch_normalization_20/Cast_1/ReadVariableOp-^batch_normalization_20/Cast_2/ReadVariableOp-^batch_normalization_20/Cast_3/ReadVariableOp+^batch_normalization_21/Cast/ReadVariableOp-^batch_normalization_21/Cast_1/ReadVariableOp-^batch_normalization_21/Cast_2/ReadVariableOp-^batch_normalization_21/Cast_3/ReadVariableOp+^batch_normalization_22/Cast/ReadVariableOp-^batch_normalization_22/Cast_1/ReadVariableOp-^batch_normalization_22/Cast_2/ReadVariableOp-^batch_normalization_22/Cast_3/ReadVariableOp^dense_15/MatMul/ReadVariableOp^dense_16/MatMul/ReadVariableOp^dense_17/MatMul/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_18/Cast/ReadVariableOp*batch_normalization_18/Cast/ReadVariableOp2\
,batch_normalization_18/Cast_1/ReadVariableOp,batch_normalization_18/Cast_1/ReadVariableOp2\
,batch_normalization_18/Cast_2/ReadVariableOp,batch_normalization_18/Cast_2/ReadVariableOp2\
,batch_normalization_18/Cast_3/ReadVariableOp,batch_normalization_18/Cast_3/ReadVariableOp2X
*batch_normalization_19/Cast/ReadVariableOp*batch_normalization_19/Cast/ReadVariableOp2\
,batch_normalization_19/Cast_1/ReadVariableOp,batch_normalization_19/Cast_1/ReadVariableOp2\
,batch_normalization_19/Cast_2/ReadVariableOp,batch_normalization_19/Cast_2/ReadVariableOp2\
,batch_normalization_19/Cast_3/ReadVariableOp,batch_normalization_19/Cast_3/ReadVariableOp2X
*batch_normalization_20/Cast/ReadVariableOp*batch_normalization_20/Cast/ReadVariableOp2\
,batch_normalization_20/Cast_1/ReadVariableOp,batch_normalization_20/Cast_1/ReadVariableOp2\
,batch_normalization_20/Cast_2/ReadVariableOp,batch_normalization_20/Cast_2/ReadVariableOp2\
,batch_normalization_20/Cast_3/ReadVariableOp,batch_normalization_20/Cast_3/ReadVariableOp2X
*batch_normalization_21/Cast/ReadVariableOp*batch_normalization_21/Cast/ReadVariableOp2\
,batch_normalization_21/Cast_1/ReadVariableOp,batch_normalization_21/Cast_1/ReadVariableOp2\
,batch_normalization_21/Cast_2/ReadVariableOp,batch_normalization_21/Cast_2/ReadVariableOp2\
,batch_normalization_21/Cast_3/ReadVariableOp,batch_normalization_21/Cast_3/ReadVariableOp2X
*batch_normalization_22/Cast/ReadVariableOp*batch_normalization_22/Cast/ReadVariableOp2\
,batch_normalization_22/Cast_1/ReadVariableOp,batch_normalization_22/Cast_1/ReadVariableOp2\
,batch_normalization_22/Cast_2/ReadVariableOp,batch_normalization_22/Cast_2/ReadVariableOp2\
,batch_normalization_22/Cast_3/ReadVariableOp,batch_normalization_22/Cast_3/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

!
_user_specified_name	input_1
ë+
Ó
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_318735

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
7__inference_batch_normalization_19_layer_call_fn_318748

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
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_3165472
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
7__inference_batch_normalization_18_layer_call_fn_318666

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
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_3163812
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
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_317107

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
â

7__inference_feed_forward_sub_net_3_layer_call_fn_318012
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
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_3175302
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
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_318945

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
È
}
)__inference_dense_16_layer_call_fn_319084

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
D__inference_dense_16_layer_call_and_return_conditional_losses_3172312
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
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_316879

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
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_318863

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
D__inference_dense_18_layer_call_and_return_conditional_losses_319119

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
D__inference_dense_17_layer_call_and_return_conditional_losses_319105

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
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_317045

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
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_319027

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
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_318899

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
÷E
ï
__inference__traced_save_319239
file_prefixR
Nsavev2_feed_forward_sub_net_3_batch_normalization_18_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_3_batch_normalization_18_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_3_batch_normalization_19_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_3_batch_normalization_19_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_3_batch_normalization_20_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_3_batch_normalization_20_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_3_batch_normalization_21_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_3_batch_normalization_21_beta_read_readvariableopR
Nsavev2_feed_forward_sub_net_3_batch_normalization_22_gamma_read_readvariableopQ
Msavev2_feed_forward_sub_net_3_batch_normalization_22_beta_read_readvariableopX
Tsavev2_feed_forward_sub_net_3_batch_normalization_18_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_3_batch_normalization_18_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_3_batch_normalization_19_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_3_batch_normalization_19_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_3_batch_normalization_20_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_3_batch_normalization_20_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_3_batch_normalization_21_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_3_batch_normalization_21_moving_variance_read_readvariableopX
Tsavev2_feed_forward_sub_net_3_batch_normalization_22_moving_mean_read_readvariableop\
Xsavev2_feed_forward_sub_net_3_batch_normalization_22_moving_variance_read_readvariableopE
Asavev2_feed_forward_sub_net_3_dense_15_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_3_dense_16_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_3_dense_17_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_3_dense_18_kernel_read_readvariableopE
Asavev2_feed_forward_sub_net_3_dense_19_kernel_read_readvariableopC
?savev2_feed_forward_sub_net_3_dense_19_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Nsavev2_feed_forward_sub_net_3_batch_normalization_18_gamma_read_readvariableopMsavev2_feed_forward_sub_net_3_batch_normalization_18_beta_read_readvariableopNsavev2_feed_forward_sub_net_3_batch_normalization_19_gamma_read_readvariableopMsavev2_feed_forward_sub_net_3_batch_normalization_19_beta_read_readvariableopNsavev2_feed_forward_sub_net_3_batch_normalization_20_gamma_read_readvariableopMsavev2_feed_forward_sub_net_3_batch_normalization_20_beta_read_readvariableopNsavev2_feed_forward_sub_net_3_batch_normalization_21_gamma_read_readvariableopMsavev2_feed_forward_sub_net_3_batch_normalization_21_beta_read_readvariableopNsavev2_feed_forward_sub_net_3_batch_normalization_22_gamma_read_readvariableopMsavev2_feed_forward_sub_net_3_batch_normalization_22_beta_read_readvariableopTsavev2_feed_forward_sub_net_3_batch_normalization_18_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_3_batch_normalization_18_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_3_batch_normalization_19_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_3_batch_normalization_19_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_3_batch_normalization_20_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_3_batch_normalization_20_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_3_batch_normalization_21_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_3_batch_normalization_21_moving_variance_read_readvariableopTsavev2_feed_forward_sub_net_3_batch_normalization_22_moving_mean_read_readvariableopXsavev2_feed_forward_sub_net_3_batch_normalization_22_moving_variance_read_readvariableopAsavev2_feed_forward_sub_net_3_dense_15_kernel_read_readvariableopAsavev2_feed_forward_sub_net_3_dense_16_kernel_read_readvariableopAsavev2_feed_forward_sub_net_3_dense_17_kernel_read_readvariableopAsavev2_feed_forward_sub_net_3_dense_18_kernel_read_readvariableopAsavev2_feed_forward_sub_net_3_dense_19_kernel_read_readvariableop?savev2_feed_forward_sub_net_3_dense_19_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
: "¨L
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
_default_save_signature
__call__
+&call_and_return_all_conditional_losses"
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
Î
-layer_metrics
regularization_losses
.metrics
	variables
/layer_regularization_losses

0layers
1non_trainable_variables
trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
3regularization_losses
4	variables
5trainable_variables
6	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ì
7axis
	gamma
beta
moving_mean
 moving_variance
8regularization_losses
9	variables
:trainable_variables
;	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ì
<axis
	gamma
beta
!moving_mean
"moving_variance
=regularization_losses
>	variables
?trainable_variables
@	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ì
Aaxis
	gamma
beta
#moving_mean
$moving_variance
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ì
Faxis
	gamma
beta
%moving_mean
&moving_variance
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
(
K	keras_api"
_tf_keras_layer
³

'kernel
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"
_tf_keras_layer
³

(kernel
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"
_tf_keras_layer
³

)kernel
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"
_tf_keras_layer
³

*kernel
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer
½

+kernel
,bias
\regularization_losses
]	variables
^trainable_variables
_	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layer
A:?
23feed_forward_sub_net_3/batch_normalization_18/gamma
@:>
22feed_forward_sub_net_3/batch_normalization_18/beta
A:?23feed_forward_sub_net_3/batch_normalization_19/gamma
@:>22feed_forward_sub_net_3/batch_normalization_19/beta
A:?23feed_forward_sub_net_3/batch_normalization_20/gamma
@:>22feed_forward_sub_net_3/batch_normalization_20/beta
A:?23feed_forward_sub_net_3/batch_normalization_21/gamma
@:>22feed_forward_sub_net_3/batch_normalization_21/beta
A:?23feed_forward_sub_net_3/batch_normalization_22/gamma
@:>22feed_forward_sub_net_3/batch_normalization_22/beta
I:G
 (29feed_forward_sub_net_3/batch_normalization_18/moving_mean
M:K
 (2=feed_forward_sub_net_3/batch_normalization_18/moving_variance
I:G (29feed_forward_sub_net_3/batch_normalization_19/moving_mean
M:K (2=feed_forward_sub_net_3/batch_normalization_19/moving_variance
I:G (29feed_forward_sub_net_3/batch_normalization_20/moving_mean
M:K (2=feed_forward_sub_net_3/batch_normalization_20/moving_variance
I:G (29feed_forward_sub_net_3/batch_normalization_21/moving_mean
M:K (2=feed_forward_sub_net_3/batch_normalization_21/moving_variance
I:G (29feed_forward_sub_net_3/batch_normalization_22/moving_mean
M:K (2=feed_forward_sub_net_3/batch_normalization_22/moving_variance
8:6
2&feed_forward_sub_net_3/dense_15/kernel
8:62&feed_forward_sub_net_3/dense_16/kernel
8:62&feed_forward_sub_net_3/dense_17/kernel
8:62&feed_forward_sub_net_3/dense_18/kernel
8:6
2&feed_forward_sub_net_3/dense_19/kernel
2:0
2$feed_forward_sub_net_3/dense_19/bias
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
°
`layer_metrics
3regularization_losses
ametrics
4	variables
blayer_regularization_losses

clayers
dnon_trainable_variables
5trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
°
elayer_metrics
8regularization_losses
fmetrics
9	variables
glayer_regularization_losses

hlayers
inon_trainable_variables
:trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
°
jlayer_metrics
=regularization_losses
kmetrics
>	variables
llayer_regularization_losses

mlayers
nnon_trainable_variables
?trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
°
olayer_metrics
Bregularization_losses
pmetrics
C	variables
qlayer_regularization_losses

rlayers
snon_trainable_variables
Dtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
°
tlayer_metrics
Gregularization_losses
umetrics
H	variables
vlayer_regularization_losses

wlayers
xnon_trainable_variables
Itrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
°
ylayer_metrics
Lregularization_losses
zmetrics
M	variables
{layer_regularization_losses

|layers
}non_trainable_variables
Ntrainable_variables
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
'
(0"
trackable_list_wrapper
³
~layer_metrics
Pregularization_losses
metrics
Q	variables
 layer_regularization_losses
layers
non_trainable_variables
Rtrainable_variables
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
µ
layer_metrics
Tregularization_losses
metrics
U	variables
 layer_regularization_losses
layers
non_trainable_variables
Vtrainable_variables
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
µ
layer_metrics
Xregularization_losses
metrics
Y	variables
 layer_regularization_losses
layers
non_trainable_variables
Ztrainable_variables
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
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
µ
layer_metrics
\regularization_losses
metrics
]	variables
 layer_regularization_losses
layers
non_trainable_variables
^trainable_variables
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
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
ÌBÉ
!__inference__wrapped_model_316357input_1"
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
7__inference_feed_forward_sub_net_3_layer_call_fn_317898
7__inference_feed_forward_sub_net_3_layer_call_fn_317955
7__inference_feed_forward_sub_net_3_layer_call_fn_318012
7__inference_feed_forward_sub_net_3_layer_call_fn_318069«
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
2þ
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_318175
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_318361
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_318467
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_318653«
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
$__inference_signature_wrapper_317841input_1"
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
¬2©
7__inference_batch_normalization_18_layer_call_fn_318666
7__inference_batch_normalization_18_layer_call_fn_318679´
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
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_318699
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_318735´
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
7__inference_batch_normalization_19_layer_call_fn_318748
7__inference_batch_normalization_19_layer_call_fn_318761´
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
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_318781
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_318817´
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
7__inference_batch_normalization_20_layer_call_fn_318830
7__inference_batch_normalization_20_layer_call_fn_318843´
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
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_318863
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_318899´
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
7__inference_batch_normalization_21_layer_call_fn_318912
7__inference_batch_normalization_21_layer_call_fn_318925´
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
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_318945
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_318981´
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
7__inference_batch_normalization_22_layer_call_fn_318994
7__inference_batch_normalization_22_layer_call_fn_319007´
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
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_319027
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_319063´
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
Ó2Ð
)__inference_dense_15_layer_call_fn_319070¢
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
D__inference_dense_15_layer_call_and_return_conditional_losses_319077¢
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
)__inference_dense_16_layer_call_fn_319084¢
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
D__inference_dense_16_layer_call_and_return_conditional_losses_319091¢
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
)__inference_dense_17_layer_call_fn_319098¢
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
D__inference_dense_17_layer_call_and_return_conditional_losses_319105¢
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
)__inference_dense_18_layer_call_fn_319112¢
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
D__inference_dense_18_layer_call_and_return_conditional_losses_319119¢
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
)__inference_dense_19_layer_call_fn_319128¢
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
D__inference_dense_19_layer_call_and_return_conditional_losses_319138¢
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
!__inference__wrapped_model_316357' (!")#$*%&+,0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ

ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
¸
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_318699b3¢0
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
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_318735b3¢0
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
7__inference_batch_normalization_18_layer_call_fn_318666U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ

7__inference_batch_normalization_18_layer_call_fn_318679U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ
¸
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_318781b 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_318817b 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_19_layer_call_fn_318748U 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_19_layer_call_fn_318761U 3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¸
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_318863b!"3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_318899b!"3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_20_layer_call_fn_318830U!"3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_20_layer_call_fn_318843U!"3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¸
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_318945b#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_318981b#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_21_layer_call_fn_318912U#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_21_layer_call_fn_318925U#$3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¸
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_319027b%&3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
R__inference_batch_normalization_22_layer_call_and_return_conditional_losses_319063b%&3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_22_layer_call_fn_318994U%&3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_22_layer_call_fn_319007U%&3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ£
D__inference_dense_15_layer_call_and_return_conditional_losses_319077['/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
)__inference_dense_15_layer_call_fn_319070N'/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ£
D__inference_dense_16_layer_call_and_return_conditional_losses_319091[(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
)__inference_dense_16_layer_call_fn_319084N(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
D__inference_dense_17_layer_call_and_return_conditional_losses_319105[)/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
)__inference_dense_17_layer_call_fn_319098N)/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
D__inference_dense_18_layer_call_and_return_conditional_losses_319119[*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
)__inference_dense_18_layer_call_fn_319112N*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_19_layer_call_and_return_conditional_losses_319138\+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 |
)__inference_dense_19_layer_call_fn_319128O+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
É
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_318175s' (!")#$*%&+,.¢+
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
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_318361s' (!")#$*%&+,.¢+
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
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_318467y' (!")#$*%&+,4¢1
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
R__inference_feed_forward_sub_net_3_layer_call_and_return_conditional_losses_318653y' (!")#$*%&+,4¢1
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
7__inference_feed_forward_sub_net_3_layer_call_fn_317898l' (!")#$*%&+,4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
¡
7__inference_feed_forward_sub_net_3_layer_call_fn_317955f' (!")#$*%&+,.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ

p 
ª "ÿÿÿÿÿÿÿÿÿ
¡
7__inference_feed_forward_sub_net_3_layer_call_fn_318012f' (!")#$*%&+,.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ
§
7__inference_feed_forward_sub_net_3_layer_call_fn_318069l' (!")#$*%&+,4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ

p
ª "ÿÿÿÿÿÿÿÿÿ
·
$__inference_signature_wrapper_317841' (!")#$*%&+,;¢8
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