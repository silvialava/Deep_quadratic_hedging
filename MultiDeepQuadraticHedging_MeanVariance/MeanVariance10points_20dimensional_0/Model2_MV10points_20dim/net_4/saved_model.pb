��
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
 �"serve*2.6.02unknown8��
�
Fnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/gamma
�
Znonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/gamma*
_output_shapes
:(*
dtype0
�
Enonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/beta
�
Ynonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/beta*
_output_shapes
:(*
dtype0
�
Fnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/gamma
�
Znonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/gamma*
_output_shapes
:<*
dtype0
�
Enonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/beta
�
Ynonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/beta*
_output_shapes
:<*
dtype0
�
Fnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/gamma
�
Znonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/gamma*
_output_shapes
:<*
dtype0
�
Enonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/beta
�
Ynonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/beta*
_output_shapes
:<*
dtype0
�
Fnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/gamma
�
Znonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/gamma*
_output_shapes
:<*
dtype0
�
Enonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/beta
�
Ynonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/beta*
_output_shapes
:<*
dtype0
�
Fnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*W
shared_nameHFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/gamma
�
Znonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/gamma/Read/ReadVariableOpReadVariableOpFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/gamma*
_output_shapes
:<*
dtype0
�
Enonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*V
shared_nameGEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/beta
�
Ynonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/beta/Read/ReadVariableOpReadVariableOpEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/beta*
_output_shapes
:<*
dtype0
�
Lnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_mean
�
`nonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_mean*
_output_shapes
:(*
dtype0
�
Pnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_variance
�
dnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_variance*
_output_shapes
:(*
dtype0
�
Lnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_mean
�
`nonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_mean*
_output_shapes
:<*
dtype0
�
Pnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_variance
�
dnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_variance*
_output_shapes
:<*
dtype0
�
Lnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_mean
�
`nonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_mean*
_output_shapes
:<*
dtype0
�
Pnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_variance
�
dnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_variance*
_output_shapes
:<*
dtype0
�
Lnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_mean
�
`nonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_mean*
_output_shapes
:<*
dtype0
�
Pnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_variance
�
dnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_variance*
_output_shapes
:<*
dtype0
�
Lnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*]
shared_nameNLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_mean
�
`nonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_mean/Read/ReadVariableOpReadVariableOpLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_mean*
_output_shapes
:<*
dtype0
�
Pnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*a
shared_nameRPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_variance
�
dnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_variance/Read/ReadVariableOpReadVariableOpPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_variance*
_output_shapes
:<*
dtype0
�
9nonshared_model_1/feed_forward_sub_net_13/dense_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(<*J
shared_name;9nonshared_model_1/feed_forward_sub_net_13/dense_65/kernel
�
Mnonshared_model_1/feed_forward_sub_net_13/dense_65/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_13/dense_65/kernel*
_output_shapes

:(<*
dtype0
�
9nonshared_model_1/feed_forward_sub_net_13/dense_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*J
shared_name;9nonshared_model_1/feed_forward_sub_net_13/dense_66/kernel
�
Mnonshared_model_1/feed_forward_sub_net_13/dense_66/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_13/dense_66/kernel*
_output_shapes

:<<*
dtype0
�
9nonshared_model_1/feed_forward_sub_net_13/dense_67/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*J
shared_name;9nonshared_model_1/feed_forward_sub_net_13/dense_67/kernel
�
Mnonshared_model_1/feed_forward_sub_net_13/dense_67/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_13/dense_67/kernel*
_output_shapes

:<<*
dtype0
�
9nonshared_model_1/feed_forward_sub_net_13/dense_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*J
shared_name;9nonshared_model_1/feed_forward_sub_net_13/dense_68/kernel
�
Mnonshared_model_1/feed_forward_sub_net_13/dense_68/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_13/dense_68/kernel*
_output_shapes

:<<*
dtype0
�
9nonshared_model_1/feed_forward_sub_net_13/dense_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<(*J
shared_name;9nonshared_model_1/feed_forward_sub_net_13/dense_69/kernel
�
Mnonshared_model_1/feed_forward_sub_net_13/dense_69/kernel/Read/ReadVariableOpReadVariableOp9nonshared_model_1/feed_forward_sub_net_13/dense_69/kernel*
_output_shapes

:<(*
dtype0
�
7nonshared_model_1/feed_forward_sub_net_13/dense_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*H
shared_name97nonshared_model_1/feed_forward_sub_net_13/dense_69/bias
�
Knonshared_model_1/feed_forward_sub_net_13/dense_69/bias/Read/ReadVariableOpReadVariableOp7nonshared_model_1/feed_forward_sub_net_13/dense_69/bias*
_output_shapes
:(*
dtype0

NoOpNoOp
�>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�>
value�>B�> B�>
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
.layer_metrics
	variables
/layer_regularization_losses
trainable_variables

0layers
1non_trainable_variables
regularization_losses
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
��
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUELnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_13/dense_65/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_13/dense_66/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_13/dense_67/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_13/dense_68/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9nonshared_model_1/feed_forward_sub_net_13/dense_69/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7nonshared_model_1/feed_forward_sub_net_13/dense_69/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
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

0
1
2
3

0
1
 
�
`layer_metrics
ametrics
3	variables
blayer_regularization_losses
4trainable_variables

clayers
dnon_trainable_variables
5regularization_losses
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
elayer_metrics
fmetrics
8	variables
glayer_regularization_losses
9trainable_variables

hlayers
inon_trainable_variables
:regularization_losses
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
jlayer_metrics
kmetrics
=	variables
llayer_regularization_losses
>trainable_variables

mlayers
nnon_trainable_variables
?regularization_losses
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
olayer_metrics
pmetrics
B	variables
qlayer_regularization_losses
Ctrainable_variables

rlayers
snon_trainable_variables
Dregularization_losses
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
tlayer_metrics
umetrics
G	variables
vlayer_regularization_losses
Htrainable_variables

wlayers
xnon_trainable_variables
Iregularization_losses
 

'0

'0
 
�
ylayer_metrics
zmetrics
L	variables
{layer_regularization_losses
Mtrainable_variables

|layers
}non_trainable_variables
Nregularization_losses

(0

(0
 
�
~layer_metrics
metrics
P	variables
 �layer_regularization_losses
Qtrainable_variables
�layers
�non_trainable_variables
Rregularization_losses

)0

)0
 
�
�layer_metrics
�metrics
T	variables
 �layer_regularization_losses
Utrainable_variables
�layers
�non_trainable_variables
Vregularization_losses

*0

*0
 
�
�layer_metrics
�metrics
X	variables
 �layer_regularization_losses
Ytrainable_variables
�layers
�non_trainable_variables
Zregularization_losses

+0
,1

+0
,1
 
�
�layer_metrics
�metrics
\	variables
 �layer_regularization_losses
]trainable_variables
�layers
�non_trainable_variables
^regularization_losses
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
:���������(*
dtype0*
shape:���������(
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Pnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_varianceFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/gammaLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_meanEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/beta9nonshared_model_1/feed_forward_sub_net_13/dense_65/kernelPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_varianceFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/gammaLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_meanEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/beta9nonshared_model_1/feed_forward_sub_net_13/dense_66/kernelPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_varianceFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/gammaLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_meanEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/beta9nonshared_model_1/feed_forward_sub_net_13/dense_67/kernelPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_varianceFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/gammaLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_meanEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/beta9nonshared_model_1/feed_forward_sub_net_13/dense_68/kernelPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_varianceFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/gammaLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_meanEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/beta9nonshared_model_1/feed_forward_sub_net_13/dense_69/kernel7nonshared_model_1/feed_forward_sub_net_13/dense_69/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_7058742
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameZnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/beta/Read/ReadVariableOpZnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/gamma/Read/ReadVariableOpYnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/beta/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_variance/Read/ReadVariableOp`nonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_mean/Read/ReadVariableOpdnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_variance/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_13/dense_65/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_13/dense_66/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_13/dense_67/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_13/dense_68/kernel/Read/ReadVariableOpMnonshared_model_1/feed_forward_sub_net_13/dense_69/kernel/Read/ReadVariableOpKnonshared_model_1/feed_forward_sub_net_13/dense_69/bias/Read/ReadVariableOpConst*'
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_7060140
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/gammaEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/betaFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/gammaEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/betaFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/gammaEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/betaFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/gammaEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/betaFnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/gammaEnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/betaLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_meanPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_varianceLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_meanPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_varianceLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_meanPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_varianceLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_meanPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_varianceLnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_meanPnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_variance9nonshared_model_1/feed_forward_sub_net_13/dense_65/kernel9nonshared_model_1/feed_forward_sub_net_13/dense_66/kernel9nonshared_model_1/feed_forward_sub_net_13/dense_67/kernel9nonshared_model_1/feed_forward_sub_net_13/dense_68/kernel9nonshared_model_1/feed_forward_sub_net_13/dense_69/kernel7nonshared_model_1/feed_forward_sub_net_13/dense_69/bias*&
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_7060228��
�
�
9__inference_feed_forward_sub_net_13_layer_call_fn_7059383
input_1
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(<
	unknown_4:<
	unknown_5:<
	unknown_6:<
	unknown_7:<
	unknown_8:<<
	unknown_9:<

unknown_10:<

unknown_11:<

unknown_12:<

unknown_13:<<

unknown_14:<

unknown_15:<

unknown_16:<

unknown_17:<

unknown_18:<<

unknown_19:<

unknown_20:<

unknown_21:<

unknown_22:<

unknown_23:<(

unknown_24:(
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
:���������(*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_70582052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������(: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������(
!
_user_specified_name	input_1
�
�
E__inference_dense_65_layer_call_and_return_conditional_losses_7059971

inputs0
matmul_readvariableop_resource:(<
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������(: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
~
*__inference_dense_68_layer_call_fn_7060020

inputs
unknown:<<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_68_layer_call_and_return_conditional_losses_70581742
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������<: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
9__inference_feed_forward_sub_net_13_layer_call_fn_7059497
x
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(<
	unknown_4:<
	unknown_5:<
	unknown_6:<
	unknown_7:<
	unknown_8:<<
	unknown_9:<

unknown_10:<

unknown_11:<

unknown_12:<

unknown_13:<<

unknown_14:<

unknown_15:<

unknown_16:<

unknown_17:<

unknown_18:<<

unknown_19:<

unknown_20:<

unknown_21:<

unknown_22:<

unknown_23:<(

unknown_24:(
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
:���������(*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_70584312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������(: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������(

_user_specified_namex
�
�
E__inference_dense_68_layer_call_and_return_conditional_losses_7060013

inputs0
matmul_readvariableop_resource:<<
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������<: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
��
�
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7058848
xF
8batch_normalization_78_batchnorm_readvariableop_resource:(J
<batch_normalization_78_batchnorm_mul_readvariableop_resource:(H
:batch_normalization_78_batchnorm_readvariableop_1_resource:(H
:batch_normalization_78_batchnorm_readvariableop_2_resource:(9
'dense_65_matmul_readvariableop_resource:(<F
8batch_normalization_79_batchnorm_readvariableop_resource:<J
<batch_normalization_79_batchnorm_mul_readvariableop_resource:<H
:batch_normalization_79_batchnorm_readvariableop_1_resource:<H
:batch_normalization_79_batchnorm_readvariableop_2_resource:<9
'dense_66_matmul_readvariableop_resource:<<F
8batch_normalization_80_batchnorm_readvariableop_resource:<J
<batch_normalization_80_batchnorm_mul_readvariableop_resource:<H
:batch_normalization_80_batchnorm_readvariableop_1_resource:<H
:batch_normalization_80_batchnorm_readvariableop_2_resource:<9
'dense_67_matmul_readvariableop_resource:<<F
8batch_normalization_81_batchnorm_readvariableop_resource:<J
<batch_normalization_81_batchnorm_mul_readvariableop_resource:<H
:batch_normalization_81_batchnorm_readvariableop_1_resource:<H
:batch_normalization_81_batchnorm_readvariableop_2_resource:<9
'dense_68_matmul_readvariableop_resource:<<F
8batch_normalization_82_batchnorm_readvariableop_resource:<J
<batch_normalization_82_batchnorm_mul_readvariableop_resource:<H
:batch_normalization_82_batchnorm_readvariableop_1_resource:<H
:batch_normalization_82_batchnorm_readvariableop_2_resource:<9
'dense_69_matmul_readvariableop_resource:<(6
(dense_69_biasadd_readvariableop_resource:(
identity��/batch_normalization_78/batchnorm/ReadVariableOp�1batch_normalization_78/batchnorm/ReadVariableOp_1�1batch_normalization_78/batchnorm/ReadVariableOp_2�3batch_normalization_78/batchnorm/mul/ReadVariableOp�/batch_normalization_79/batchnorm/ReadVariableOp�1batch_normalization_79/batchnorm/ReadVariableOp_1�1batch_normalization_79/batchnorm/ReadVariableOp_2�3batch_normalization_79/batchnorm/mul/ReadVariableOp�/batch_normalization_80/batchnorm/ReadVariableOp�1batch_normalization_80/batchnorm/ReadVariableOp_1�1batch_normalization_80/batchnorm/ReadVariableOp_2�3batch_normalization_80/batchnorm/mul/ReadVariableOp�/batch_normalization_81/batchnorm/ReadVariableOp�1batch_normalization_81/batchnorm/ReadVariableOp_1�1batch_normalization_81/batchnorm/ReadVariableOp_2�3batch_normalization_81/batchnorm/mul/ReadVariableOp�/batch_normalization_82/batchnorm/ReadVariableOp�1batch_normalization_82/batchnorm/ReadVariableOp_1�1batch_normalization_82/batchnorm/ReadVariableOp_2�3batch_normalization_82/batchnorm/mul/ReadVariableOp�dense_65/MatMul/ReadVariableOp�dense_66/MatMul/ReadVariableOp�dense_67/MatMul/ReadVariableOp�dense_68/MatMul/ReadVariableOp�dense_69/BiasAdd/ReadVariableOp�dense_69/MatMul/ReadVariableOp�
/batch_normalization_78/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_78_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype021
/batch_normalization_78/batchnorm/ReadVariableOp�
&batch_normalization_78/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_78/batchnorm/add/y�
$batch_normalization_78/batchnorm/addAddV27batch_normalization_78/batchnorm/ReadVariableOp:value:0/batch_normalization_78/batchnorm/add/y:output:0*
T0*
_output_shapes
:(2&
$batch_normalization_78/batchnorm/add�
&batch_normalization_78/batchnorm/RsqrtRsqrt(batch_normalization_78/batchnorm/add:z:0*
T0*
_output_shapes
:(2(
&batch_normalization_78/batchnorm/Rsqrt�
3batch_normalization_78/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_78_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype025
3batch_normalization_78/batchnorm/mul/ReadVariableOp�
$batch_normalization_78/batchnorm/mulMul*batch_normalization_78/batchnorm/Rsqrt:y:0;batch_normalization_78/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2&
$batch_normalization_78/batchnorm/mul�
&batch_normalization_78/batchnorm/mul_1Mulx(batch_normalization_78/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2(
&batch_normalization_78/batchnorm/mul_1�
1batch_normalization_78/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_78_batchnorm_readvariableop_1_resource*
_output_shapes
:(*
dtype023
1batch_normalization_78/batchnorm/ReadVariableOp_1�
&batch_normalization_78/batchnorm/mul_2Mul9batch_normalization_78/batchnorm/ReadVariableOp_1:value:0(batch_normalization_78/batchnorm/mul:z:0*
T0*
_output_shapes
:(2(
&batch_normalization_78/batchnorm/mul_2�
1batch_normalization_78/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_78_batchnorm_readvariableop_2_resource*
_output_shapes
:(*
dtype023
1batch_normalization_78/batchnorm/ReadVariableOp_2�
$batch_normalization_78/batchnorm/subSub9batch_normalization_78/batchnorm/ReadVariableOp_2:value:0*batch_normalization_78/batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2&
$batch_normalization_78/batchnorm/sub�
&batch_normalization_78/batchnorm/add_1AddV2*batch_normalization_78/batchnorm/mul_1:z:0(batch_normalization_78/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2(
&batch_normalization_78/batchnorm/add_1�
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02 
dense_65/MatMul/ReadVariableOp�
dense_65/MatMulMatMul*batch_normalization_78/batchnorm/add_1:z:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_65/MatMul�
/batch_normalization_79/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_79_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_79/batchnorm/ReadVariableOp�
&batch_normalization_79/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_79/batchnorm/add/y�
$batch_normalization_79/batchnorm/addAddV27batch_normalization_79/batchnorm/ReadVariableOp:value:0/batch_normalization_79/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_79/batchnorm/add�
&batch_normalization_79/batchnorm/RsqrtRsqrt(batch_normalization_79/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_79/batchnorm/Rsqrt�
3batch_normalization_79/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_79_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_79/batchnorm/mul/ReadVariableOp�
$batch_normalization_79/batchnorm/mulMul*batch_normalization_79/batchnorm/Rsqrt:y:0;batch_normalization_79/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_79/batchnorm/mul�
&batch_normalization_79/batchnorm/mul_1Muldense_65/MatMul:product:0(batch_normalization_79/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_79/batchnorm/mul_1�
1batch_normalization_79/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_79_batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype023
1batch_normalization_79/batchnorm/ReadVariableOp_1�
&batch_normalization_79/batchnorm/mul_2Mul9batch_normalization_79/batchnorm/ReadVariableOp_1:value:0(batch_normalization_79/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_79/batchnorm/mul_2�
1batch_normalization_79/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_79_batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype023
1batch_normalization_79/batchnorm/ReadVariableOp_2�
$batch_normalization_79/batchnorm/subSub9batch_normalization_79/batchnorm/ReadVariableOp_2:value:0*batch_normalization_79/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_79/batchnorm/sub�
&batch_normalization_79/batchnorm/add_1AddV2*batch_normalization_79/batchnorm/mul_1:z:0(batch_normalization_79/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_79/batchnorm/add_1r
ReluRelu*batch_normalization_79/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu�
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02 
dense_66/MatMul/ReadVariableOp�
dense_66/MatMulMatMulRelu:activations:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_66/MatMul�
/batch_normalization_80/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_80_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_80/batchnorm/ReadVariableOp�
&batch_normalization_80/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_80/batchnorm/add/y�
$batch_normalization_80/batchnorm/addAddV27batch_normalization_80/batchnorm/ReadVariableOp:value:0/batch_normalization_80/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_80/batchnorm/add�
&batch_normalization_80/batchnorm/RsqrtRsqrt(batch_normalization_80/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_80/batchnorm/Rsqrt�
3batch_normalization_80/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_80_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_80/batchnorm/mul/ReadVariableOp�
$batch_normalization_80/batchnorm/mulMul*batch_normalization_80/batchnorm/Rsqrt:y:0;batch_normalization_80/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_80/batchnorm/mul�
&batch_normalization_80/batchnorm/mul_1Muldense_66/MatMul:product:0(batch_normalization_80/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_80/batchnorm/mul_1�
1batch_normalization_80/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_80_batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype023
1batch_normalization_80/batchnorm/ReadVariableOp_1�
&batch_normalization_80/batchnorm/mul_2Mul9batch_normalization_80/batchnorm/ReadVariableOp_1:value:0(batch_normalization_80/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_80/batchnorm/mul_2�
1batch_normalization_80/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_80_batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype023
1batch_normalization_80/batchnorm/ReadVariableOp_2�
$batch_normalization_80/batchnorm/subSub9batch_normalization_80/batchnorm/ReadVariableOp_2:value:0*batch_normalization_80/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_80/batchnorm/sub�
&batch_normalization_80/batchnorm/add_1AddV2*batch_normalization_80/batchnorm/mul_1:z:0(batch_normalization_80/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_80/batchnorm/add_1v
Relu_1Relu*batch_normalization_80/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu_1�
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02 
dense_67/MatMul/ReadVariableOp�
dense_67/MatMulMatMulRelu_1:activations:0&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_67/MatMul�
/batch_normalization_81/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_81_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_81/batchnorm/ReadVariableOp�
&batch_normalization_81/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_81/batchnorm/add/y�
$batch_normalization_81/batchnorm/addAddV27batch_normalization_81/batchnorm/ReadVariableOp:value:0/batch_normalization_81/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_81/batchnorm/add�
&batch_normalization_81/batchnorm/RsqrtRsqrt(batch_normalization_81/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_81/batchnorm/Rsqrt�
3batch_normalization_81/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_81_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_81/batchnorm/mul/ReadVariableOp�
$batch_normalization_81/batchnorm/mulMul*batch_normalization_81/batchnorm/Rsqrt:y:0;batch_normalization_81/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_81/batchnorm/mul�
&batch_normalization_81/batchnorm/mul_1Muldense_67/MatMul:product:0(batch_normalization_81/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_81/batchnorm/mul_1�
1batch_normalization_81/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_81_batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype023
1batch_normalization_81/batchnorm/ReadVariableOp_1�
&batch_normalization_81/batchnorm/mul_2Mul9batch_normalization_81/batchnorm/ReadVariableOp_1:value:0(batch_normalization_81/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_81/batchnorm/mul_2�
1batch_normalization_81/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_81_batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype023
1batch_normalization_81/batchnorm/ReadVariableOp_2�
$batch_normalization_81/batchnorm/subSub9batch_normalization_81/batchnorm/ReadVariableOp_2:value:0*batch_normalization_81/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_81/batchnorm/sub�
&batch_normalization_81/batchnorm/add_1AddV2*batch_normalization_81/batchnorm/mul_1:z:0(batch_normalization_81/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_81/batchnorm/add_1v
Relu_2Relu*batch_normalization_81/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu_2�
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02 
dense_68/MatMul/ReadVariableOp�
dense_68/MatMulMatMulRelu_2:activations:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_68/MatMul�
/batch_normalization_82/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_82_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_82/batchnorm/ReadVariableOp�
&batch_normalization_82/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_82/batchnorm/add/y�
$batch_normalization_82/batchnorm/addAddV27batch_normalization_82/batchnorm/ReadVariableOp:value:0/batch_normalization_82/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_82/batchnorm/add�
&batch_normalization_82/batchnorm/RsqrtRsqrt(batch_normalization_82/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_82/batchnorm/Rsqrt�
3batch_normalization_82/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_82_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_82/batchnorm/mul/ReadVariableOp�
$batch_normalization_82/batchnorm/mulMul*batch_normalization_82/batchnorm/Rsqrt:y:0;batch_normalization_82/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_82/batchnorm/mul�
&batch_normalization_82/batchnorm/mul_1Muldense_68/MatMul:product:0(batch_normalization_82/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_82/batchnorm/mul_1�
1batch_normalization_82/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_82_batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype023
1batch_normalization_82/batchnorm/ReadVariableOp_1�
&batch_normalization_82/batchnorm/mul_2Mul9batch_normalization_82/batchnorm/ReadVariableOp_1:value:0(batch_normalization_82/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_82/batchnorm/mul_2�
1batch_normalization_82/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_82_batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype023
1batch_normalization_82/batchnorm/ReadVariableOp_2�
$batch_normalization_82/batchnorm/subSub9batch_normalization_82/batchnorm/ReadVariableOp_2:value:0*batch_normalization_82/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_82/batchnorm/sub�
&batch_normalization_82/batchnorm/add_1AddV2*batch_normalization_82/batchnorm/mul_1:z:0(batch_normalization_82/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_82/batchnorm/add_1v
Relu_3Relu*batch_normalization_82/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu_3�
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource*
_output_shapes

:<(*
dtype02 
dense_69/MatMul/ReadVariableOp�
dense_69/MatMulMatMulRelu_3:activations:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_69/MatMul�
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02!
dense_69/BiasAdd/ReadVariableOp�
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_69/BiasAddt
IdentityIdentitydense_69/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity�

NoOpNoOp0^batch_normalization_78/batchnorm/ReadVariableOp2^batch_normalization_78/batchnorm/ReadVariableOp_12^batch_normalization_78/batchnorm/ReadVariableOp_24^batch_normalization_78/batchnorm/mul/ReadVariableOp0^batch_normalization_79/batchnorm/ReadVariableOp2^batch_normalization_79/batchnorm/ReadVariableOp_12^batch_normalization_79/batchnorm/ReadVariableOp_24^batch_normalization_79/batchnorm/mul/ReadVariableOp0^batch_normalization_80/batchnorm/ReadVariableOp2^batch_normalization_80/batchnorm/ReadVariableOp_12^batch_normalization_80/batchnorm/ReadVariableOp_24^batch_normalization_80/batchnorm/mul/ReadVariableOp0^batch_normalization_81/batchnorm/ReadVariableOp2^batch_normalization_81/batchnorm/ReadVariableOp_12^batch_normalization_81/batchnorm/ReadVariableOp_24^batch_normalization_81/batchnorm/mul/ReadVariableOp0^batch_normalization_82/batchnorm/ReadVariableOp2^batch_normalization_82/batchnorm/ReadVariableOp_12^batch_normalization_82/batchnorm/ReadVariableOp_24^batch_normalization_82/batchnorm/mul/ReadVariableOp^dense_65/MatMul/ReadVariableOp^dense_66/MatMul/ReadVariableOp^dense_67/MatMul/ReadVariableOp^dense_68/MatMul/ReadVariableOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������(: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_78/batchnorm/ReadVariableOp/batch_normalization_78/batchnorm/ReadVariableOp2f
1batch_normalization_78/batchnorm/ReadVariableOp_11batch_normalization_78/batchnorm/ReadVariableOp_12f
1batch_normalization_78/batchnorm/ReadVariableOp_21batch_normalization_78/batchnorm/ReadVariableOp_22j
3batch_normalization_78/batchnorm/mul/ReadVariableOp3batch_normalization_78/batchnorm/mul/ReadVariableOp2b
/batch_normalization_79/batchnorm/ReadVariableOp/batch_normalization_79/batchnorm/ReadVariableOp2f
1batch_normalization_79/batchnorm/ReadVariableOp_11batch_normalization_79/batchnorm/ReadVariableOp_12f
1batch_normalization_79/batchnorm/ReadVariableOp_21batch_normalization_79/batchnorm/ReadVariableOp_22j
3batch_normalization_79/batchnorm/mul/ReadVariableOp3batch_normalization_79/batchnorm/mul/ReadVariableOp2b
/batch_normalization_80/batchnorm/ReadVariableOp/batch_normalization_80/batchnorm/ReadVariableOp2f
1batch_normalization_80/batchnorm/ReadVariableOp_11batch_normalization_80/batchnorm/ReadVariableOp_12f
1batch_normalization_80/batchnorm/ReadVariableOp_21batch_normalization_80/batchnorm/ReadVariableOp_22j
3batch_normalization_80/batchnorm/mul/ReadVariableOp3batch_normalization_80/batchnorm/mul/ReadVariableOp2b
/batch_normalization_81/batchnorm/ReadVariableOp/batch_normalization_81/batchnorm/ReadVariableOp2f
1batch_normalization_81/batchnorm/ReadVariableOp_11batch_normalization_81/batchnorm/ReadVariableOp_12f
1batch_normalization_81/batchnorm/ReadVariableOp_21batch_normalization_81/batchnorm/ReadVariableOp_22j
3batch_normalization_81/batchnorm/mul/ReadVariableOp3batch_normalization_81/batchnorm/mul/ReadVariableOp2b
/batch_normalization_82/batchnorm/ReadVariableOp/batch_normalization_82/batchnorm/ReadVariableOp2f
1batch_normalization_82/batchnorm/ReadVariableOp_11batch_normalization_82/batchnorm/ReadVariableOp_12f
1batch_normalization_82/batchnorm/ReadVariableOp_21batch_normalization_82/batchnorm/ReadVariableOp_22j
3batch_normalization_82/batchnorm/mul/ReadVariableOp3batch_normalization_82/batchnorm/mul/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������(

_user_specified_namex
�
�
S__inference_batch_normalization_80_layer_call_and_return_conditional_losses_7059738

inputs/
!batchnorm_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:<1
#batchnorm_readvariableop_1_resource:<1
#batchnorm_readvariableop_2_resource:<
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_80_layer_call_fn_7059787

inputs
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_80_layer_call_and_return_conditional_losses_70576142
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_80_layer_call_and_return_conditional_losses_7057614

inputs/
!batchnorm_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:<1
#batchnorm_readvariableop_1_resource:<1
#batchnorm_readvariableop_2_resource:<
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�E
�
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7058205
x,
batch_normalization_78_7058095:(,
batch_normalization_78_7058097:(,
batch_normalization_78_7058099:(,
batch_normalization_78_7058101:("
dense_65_7058112:(<,
batch_normalization_79_7058115:<,
batch_normalization_79_7058117:<,
batch_normalization_79_7058119:<,
batch_normalization_79_7058121:<"
dense_66_7058133:<<,
batch_normalization_80_7058136:<,
batch_normalization_80_7058138:<,
batch_normalization_80_7058140:<,
batch_normalization_80_7058142:<"
dense_67_7058154:<<,
batch_normalization_81_7058157:<,
batch_normalization_81_7058159:<,
batch_normalization_81_7058161:<,
batch_normalization_81_7058163:<"
dense_68_7058175:<<,
batch_normalization_82_7058178:<,
batch_normalization_82_7058180:<,
batch_normalization_82_7058182:<,
batch_normalization_82_7058184:<"
dense_69_7058199:<(
dense_69_7058201:(
identity��.batch_normalization_78/StatefulPartitionedCall�.batch_normalization_79/StatefulPartitionedCall�.batch_normalization_80/StatefulPartitionedCall�.batch_normalization_81/StatefulPartitionedCall�.batch_normalization_82/StatefulPartitionedCall� dense_65/StatefulPartitionedCall� dense_66/StatefulPartitionedCall� dense_67/StatefulPartitionedCall� dense_68/StatefulPartitionedCall� dense_69/StatefulPartitionedCall�
.batch_normalization_78/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_78_7058095batch_normalization_78_7058097batch_normalization_78_7058099batch_normalization_78_7058101*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_78_layer_call_and_return_conditional_losses_705728220
.batch_normalization_78/StatefulPartitionedCall�
 dense_65/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_78/StatefulPartitionedCall:output:0dense_65_7058112*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_70581112"
 dense_65/StatefulPartitionedCall�
.batch_normalization_79/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0batch_normalization_79_7058115batch_normalization_79_7058117batch_normalization_79_7058119batch_normalization_79_7058121*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_79_layer_call_and_return_conditional_losses_705744820
.batch_normalization_79/StatefulPartitionedCall
ReluRelu7batch_normalization_79/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������<2
Relu�
 dense_66/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_66_7058133*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_66_layer_call_and_return_conditional_losses_70581322"
 dense_66/StatefulPartitionedCall�
.batch_normalization_80/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0batch_normalization_80_7058136batch_normalization_80_7058138batch_normalization_80_7058140batch_normalization_80_7058142*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_80_layer_call_and_return_conditional_losses_705761420
.batch_normalization_80/StatefulPartitionedCall�
Relu_1Relu7batch_normalization_80/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������<2
Relu_1�
 dense_67/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_67_7058154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_67_layer_call_and_return_conditional_losses_70581532"
 dense_67/StatefulPartitionedCall�
.batch_normalization_81/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0batch_normalization_81_7058157batch_normalization_81_7058159batch_normalization_81_7058161batch_normalization_81_7058163*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_81_layer_call_and_return_conditional_losses_705778020
.batch_normalization_81/StatefulPartitionedCall�
Relu_2Relu7batch_normalization_81/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������<2
Relu_2�
 dense_68/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_68_7058175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_68_layer_call_and_return_conditional_losses_70581742"
 dense_68/StatefulPartitionedCall�
.batch_normalization_82/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0batch_normalization_82_7058178batch_normalization_82_7058180batch_normalization_82_7058182batch_normalization_82_7058184*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_82_layer_call_and_return_conditional_losses_705794620
.batch_normalization_82/StatefulPartitionedCall�
Relu_3Relu7batch_normalization_82/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������<2
Relu_3�
 dense_69/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_69_7058199dense_69_7058201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_69_layer_call_and_return_conditional_losses_70581982"
 dense_69/StatefulPartitionedCall�
IdentityIdentity)dense_69/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity�
NoOpNoOp/^batch_normalization_78/StatefulPartitionedCall/^batch_normalization_79/StatefulPartitionedCall/^batch_normalization_80/StatefulPartitionedCall/^batch_normalization_81/StatefulPartitionedCall/^batch_normalization_82/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������(: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_78/StatefulPartitionedCall.batch_normalization_78/StatefulPartitionedCall2`
.batch_normalization_79/StatefulPartitionedCall.batch_normalization_79/StatefulPartitionedCall2`
.batch_normalization_80/StatefulPartitionedCall.batch_normalization_80/StatefulPartitionedCall2`
.batch_normalization_81/StatefulPartitionedCall.batch_normalization_81/StatefulPartitionedCall2`
.batch_normalization_82/StatefulPartitionedCall.batch_normalization_82/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall:J F
'
_output_shapes
:���������(

_user_specified_namex
�
�
E__inference_dense_66_layer_call_and_return_conditional_losses_7058132

inputs0
matmul_readvariableop_resource:<<
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������<: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
E__inference_dense_67_layer_call_and_return_conditional_losses_7058153

inputs0
matmul_readvariableop_resource:<<
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������<: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_78_layer_call_and_return_conditional_losses_7059610

inputs5
'assignmovingavg_readvariableop_resource:(7
)assignmovingavg_1_readvariableop_resource:(3
%batchnorm_mul_readvariableop_resource:(/
!batchnorm_readvariableop_resource:(
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:(*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:(2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������(2
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

:(*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:(*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:(*
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
:(*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:(2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:(2
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
:(*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:(2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:(2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:(2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:(2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:(2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������(: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
*__inference_dense_69_layer_call_fn_7060039

inputs
unknown:<(
	unknown_0:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_69_layer_call_and_return_conditional_losses_70581982
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������<: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_80_layer_call_fn_7059800

inputs
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_80_layer_call_and_return_conditional_losses_70576762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
��
�"
"__inference__wrapped_model_7057258
input_1^
Pfeed_forward_sub_net_13_batch_normalization_78_batchnorm_readvariableop_resource:(b
Tfeed_forward_sub_net_13_batch_normalization_78_batchnorm_mul_readvariableop_resource:(`
Rfeed_forward_sub_net_13_batch_normalization_78_batchnorm_readvariableop_1_resource:(`
Rfeed_forward_sub_net_13_batch_normalization_78_batchnorm_readvariableop_2_resource:(Q
?feed_forward_sub_net_13_dense_65_matmul_readvariableop_resource:(<^
Pfeed_forward_sub_net_13_batch_normalization_79_batchnorm_readvariableop_resource:<b
Tfeed_forward_sub_net_13_batch_normalization_79_batchnorm_mul_readvariableop_resource:<`
Rfeed_forward_sub_net_13_batch_normalization_79_batchnorm_readvariableop_1_resource:<`
Rfeed_forward_sub_net_13_batch_normalization_79_batchnorm_readvariableop_2_resource:<Q
?feed_forward_sub_net_13_dense_66_matmul_readvariableop_resource:<<^
Pfeed_forward_sub_net_13_batch_normalization_80_batchnorm_readvariableop_resource:<b
Tfeed_forward_sub_net_13_batch_normalization_80_batchnorm_mul_readvariableop_resource:<`
Rfeed_forward_sub_net_13_batch_normalization_80_batchnorm_readvariableop_1_resource:<`
Rfeed_forward_sub_net_13_batch_normalization_80_batchnorm_readvariableop_2_resource:<Q
?feed_forward_sub_net_13_dense_67_matmul_readvariableop_resource:<<^
Pfeed_forward_sub_net_13_batch_normalization_81_batchnorm_readvariableop_resource:<b
Tfeed_forward_sub_net_13_batch_normalization_81_batchnorm_mul_readvariableop_resource:<`
Rfeed_forward_sub_net_13_batch_normalization_81_batchnorm_readvariableop_1_resource:<`
Rfeed_forward_sub_net_13_batch_normalization_81_batchnorm_readvariableop_2_resource:<Q
?feed_forward_sub_net_13_dense_68_matmul_readvariableop_resource:<<^
Pfeed_forward_sub_net_13_batch_normalization_82_batchnorm_readvariableop_resource:<b
Tfeed_forward_sub_net_13_batch_normalization_82_batchnorm_mul_readvariableop_resource:<`
Rfeed_forward_sub_net_13_batch_normalization_82_batchnorm_readvariableop_1_resource:<`
Rfeed_forward_sub_net_13_batch_normalization_82_batchnorm_readvariableop_2_resource:<Q
?feed_forward_sub_net_13_dense_69_matmul_readvariableop_resource:<(N
@feed_forward_sub_net_13_dense_69_biasadd_readvariableop_resource:(
identity��Gfeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp�Ifeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_1�Ifeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_2�Kfeed_forward_sub_net_13/batch_normalization_78/batchnorm/mul/ReadVariableOp�Gfeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp�Ifeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_1�Ifeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_2�Kfeed_forward_sub_net_13/batch_normalization_79/batchnorm/mul/ReadVariableOp�Gfeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp�Ifeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_1�Ifeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_2�Kfeed_forward_sub_net_13/batch_normalization_80/batchnorm/mul/ReadVariableOp�Gfeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp�Ifeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_1�Ifeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_2�Kfeed_forward_sub_net_13/batch_normalization_81/batchnorm/mul/ReadVariableOp�Gfeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp�Ifeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_1�Ifeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_2�Kfeed_forward_sub_net_13/batch_normalization_82/batchnorm/mul/ReadVariableOp�6feed_forward_sub_net_13/dense_65/MatMul/ReadVariableOp�6feed_forward_sub_net_13/dense_66/MatMul/ReadVariableOp�6feed_forward_sub_net_13/dense_67/MatMul/ReadVariableOp�6feed_forward_sub_net_13/dense_68/MatMul/ReadVariableOp�7feed_forward_sub_net_13/dense_69/BiasAdd/ReadVariableOp�6feed_forward_sub_net_13/dense_69/MatMul/ReadVariableOp�
Gfeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_13_batch_normalization_78_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02I
Gfeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp�
>feed_forward_sub_net_13/batch_normalization_78/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2@
>feed_forward_sub_net_13/batch_normalization_78/batchnorm/add/y�
<feed_forward_sub_net_13/batch_normalization_78/batchnorm/addAddV2Ofeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_13/batch_normalization_78/batchnorm/add/y:output:0*
T0*
_output_shapes
:(2>
<feed_forward_sub_net_13/batch_normalization_78/batchnorm/add�
>feed_forward_sub_net_13/batch_normalization_78/batchnorm/RsqrtRsqrt@feed_forward_sub_net_13/batch_normalization_78/batchnorm/add:z:0*
T0*
_output_shapes
:(2@
>feed_forward_sub_net_13/batch_normalization_78/batchnorm/Rsqrt�
Kfeed_forward_sub_net_13/batch_normalization_78/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_13_batch_normalization_78_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02M
Kfeed_forward_sub_net_13/batch_normalization_78/batchnorm/mul/ReadVariableOp�
<feed_forward_sub_net_13/batch_normalization_78/batchnorm/mulMulBfeed_forward_sub_net_13/batch_normalization_78/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_13/batch_normalization_78/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2>
<feed_forward_sub_net_13/batch_normalization_78/batchnorm/mul�
>feed_forward_sub_net_13/batch_normalization_78/batchnorm/mul_1Mulinput_1@feed_forward_sub_net_13/batch_normalization_78/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2@
>feed_forward_sub_net_13/batch_normalization_78/batchnorm/mul_1�
Ifeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_13_batch_normalization_78_batchnorm_readvariableop_1_resource*
_output_shapes
:(*
dtype02K
Ifeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_1�
>feed_forward_sub_net_13/batch_normalization_78/batchnorm/mul_2MulQfeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_13/batch_normalization_78/batchnorm/mul:z:0*
T0*
_output_shapes
:(2@
>feed_forward_sub_net_13/batch_normalization_78/batchnorm/mul_2�
Ifeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_13_batch_normalization_78_batchnorm_readvariableop_2_resource*
_output_shapes
:(*
dtype02K
Ifeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_2�
<feed_forward_sub_net_13/batch_normalization_78/batchnorm/subSubQfeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_13/batch_normalization_78/batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2>
<feed_forward_sub_net_13/batch_normalization_78/batchnorm/sub�
>feed_forward_sub_net_13/batch_normalization_78/batchnorm/add_1AddV2Bfeed_forward_sub_net_13/batch_normalization_78/batchnorm/mul_1:z:0@feed_forward_sub_net_13/batch_normalization_78/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2@
>feed_forward_sub_net_13/batch_normalization_78/batchnorm/add_1�
6feed_forward_sub_net_13/dense_65/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_13_dense_65_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype028
6feed_forward_sub_net_13/dense_65/MatMul/ReadVariableOp�
'feed_forward_sub_net_13/dense_65/MatMulMatMulBfeed_forward_sub_net_13/batch_normalization_78/batchnorm/add_1:z:0>feed_forward_sub_net_13/dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2)
'feed_forward_sub_net_13/dense_65/MatMul�
Gfeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_13_batch_normalization_79_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02I
Gfeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp�
>feed_forward_sub_net_13/batch_normalization_79/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2@
>feed_forward_sub_net_13/batch_normalization_79/batchnorm/add/y�
<feed_forward_sub_net_13/batch_normalization_79/batchnorm/addAddV2Ofeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_13/batch_normalization_79/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2>
<feed_forward_sub_net_13/batch_normalization_79/batchnorm/add�
>feed_forward_sub_net_13/batch_normalization_79/batchnorm/RsqrtRsqrt@feed_forward_sub_net_13/batch_normalization_79/batchnorm/add:z:0*
T0*
_output_shapes
:<2@
>feed_forward_sub_net_13/batch_normalization_79/batchnorm/Rsqrt�
Kfeed_forward_sub_net_13/batch_normalization_79/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_13_batch_normalization_79_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02M
Kfeed_forward_sub_net_13/batch_normalization_79/batchnorm/mul/ReadVariableOp�
<feed_forward_sub_net_13/batch_normalization_79/batchnorm/mulMulBfeed_forward_sub_net_13/batch_normalization_79/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_13/batch_normalization_79/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2>
<feed_forward_sub_net_13/batch_normalization_79/batchnorm/mul�
>feed_forward_sub_net_13/batch_normalization_79/batchnorm/mul_1Mul1feed_forward_sub_net_13/dense_65/MatMul:product:0@feed_forward_sub_net_13/batch_normalization_79/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2@
>feed_forward_sub_net_13/batch_normalization_79/batchnorm/mul_1�
Ifeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_13_batch_normalization_79_batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype02K
Ifeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_1�
>feed_forward_sub_net_13/batch_normalization_79/batchnorm/mul_2MulQfeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_13/batch_normalization_79/batchnorm/mul:z:0*
T0*
_output_shapes
:<2@
>feed_forward_sub_net_13/batch_normalization_79/batchnorm/mul_2�
Ifeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_13_batch_normalization_79_batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype02K
Ifeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_2�
<feed_forward_sub_net_13/batch_normalization_79/batchnorm/subSubQfeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_13/batch_normalization_79/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2>
<feed_forward_sub_net_13/batch_normalization_79/batchnorm/sub�
>feed_forward_sub_net_13/batch_normalization_79/batchnorm/add_1AddV2Bfeed_forward_sub_net_13/batch_normalization_79/batchnorm/mul_1:z:0@feed_forward_sub_net_13/batch_normalization_79/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2@
>feed_forward_sub_net_13/batch_normalization_79/batchnorm/add_1�
feed_forward_sub_net_13/ReluReluBfeed_forward_sub_net_13/batch_normalization_79/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
feed_forward_sub_net_13/Relu�
6feed_forward_sub_net_13/dense_66/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_13_dense_66_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype028
6feed_forward_sub_net_13/dense_66/MatMul/ReadVariableOp�
'feed_forward_sub_net_13/dense_66/MatMulMatMul*feed_forward_sub_net_13/Relu:activations:0>feed_forward_sub_net_13/dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2)
'feed_forward_sub_net_13/dense_66/MatMul�
Gfeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_13_batch_normalization_80_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02I
Gfeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp�
>feed_forward_sub_net_13/batch_normalization_80/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2@
>feed_forward_sub_net_13/batch_normalization_80/batchnorm/add/y�
<feed_forward_sub_net_13/batch_normalization_80/batchnorm/addAddV2Ofeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_13/batch_normalization_80/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2>
<feed_forward_sub_net_13/batch_normalization_80/batchnorm/add�
>feed_forward_sub_net_13/batch_normalization_80/batchnorm/RsqrtRsqrt@feed_forward_sub_net_13/batch_normalization_80/batchnorm/add:z:0*
T0*
_output_shapes
:<2@
>feed_forward_sub_net_13/batch_normalization_80/batchnorm/Rsqrt�
Kfeed_forward_sub_net_13/batch_normalization_80/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_13_batch_normalization_80_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02M
Kfeed_forward_sub_net_13/batch_normalization_80/batchnorm/mul/ReadVariableOp�
<feed_forward_sub_net_13/batch_normalization_80/batchnorm/mulMulBfeed_forward_sub_net_13/batch_normalization_80/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_13/batch_normalization_80/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2>
<feed_forward_sub_net_13/batch_normalization_80/batchnorm/mul�
>feed_forward_sub_net_13/batch_normalization_80/batchnorm/mul_1Mul1feed_forward_sub_net_13/dense_66/MatMul:product:0@feed_forward_sub_net_13/batch_normalization_80/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2@
>feed_forward_sub_net_13/batch_normalization_80/batchnorm/mul_1�
Ifeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_13_batch_normalization_80_batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype02K
Ifeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_1�
>feed_forward_sub_net_13/batch_normalization_80/batchnorm/mul_2MulQfeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_13/batch_normalization_80/batchnorm/mul:z:0*
T0*
_output_shapes
:<2@
>feed_forward_sub_net_13/batch_normalization_80/batchnorm/mul_2�
Ifeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_13_batch_normalization_80_batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype02K
Ifeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_2�
<feed_forward_sub_net_13/batch_normalization_80/batchnorm/subSubQfeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_13/batch_normalization_80/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2>
<feed_forward_sub_net_13/batch_normalization_80/batchnorm/sub�
>feed_forward_sub_net_13/batch_normalization_80/batchnorm/add_1AddV2Bfeed_forward_sub_net_13/batch_normalization_80/batchnorm/mul_1:z:0@feed_forward_sub_net_13/batch_normalization_80/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2@
>feed_forward_sub_net_13/batch_normalization_80/batchnorm/add_1�
feed_forward_sub_net_13/Relu_1ReluBfeed_forward_sub_net_13/batch_normalization_80/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2 
feed_forward_sub_net_13/Relu_1�
6feed_forward_sub_net_13/dense_67/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_13_dense_67_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype028
6feed_forward_sub_net_13/dense_67/MatMul/ReadVariableOp�
'feed_forward_sub_net_13/dense_67/MatMulMatMul,feed_forward_sub_net_13/Relu_1:activations:0>feed_forward_sub_net_13/dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2)
'feed_forward_sub_net_13/dense_67/MatMul�
Gfeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_13_batch_normalization_81_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02I
Gfeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp�
>feed_forward_sub_net_13/batch_normalization_81/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2@
>feed_forward_sub_net_13/batch_normalization_81/batchnorm/add/y�
<feed_forward_sub_net_13/batch_normalization_81/batchnorm/addAddV2Ofeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_13/batch_normalization_81/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2>
<feed_forward_sub_net_13/batch_normalization_81/batchnorm/add�
>feed_forward_sub_net_13/batch_normalization_81/batchnorm/RsqrtRsqrt@feed_forward_sub_net_13/batch_normalization_81/batchnorm/add:z:0*
T0*
_output_shapes
:<2@
>feed_forward_sub_net_13/batch_normalization_81/batchnorm/Rsqrt�
Kfeed_forward_sub_net_13/batch_normalization_81/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_13_batch_normalization_81_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02M
Kfeed_forward_sub_net_13/batch_normalization_81/batchnorm/mul/ReadVariableOp�
<feed_forward_sub_net_13/batch_normalization_81/batchnorm/mulMulBfeed_forward_sub_net_13/batch_normalization_81/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_13/batch_normalization_81/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2>
<feed_forward_sub_net_13/batch_normalization_81/batchnorm/mul�
>feed_forward_sub_net_13/batch_normalization_81/batchnorm/mul_1Mul1feed_forward_sub_net_13/dense_67/MatMul:product:0@feed_forward_sub_net_13/batch_normalization_81/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2@
>feed_forward_sub_net_13/batch_normalization_81/batchnorm/mul_1�
Ifeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_13_batch_normalization_81_batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype02K
Ifeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_1�
>feed_forward_sub_net_13/batch_normalization_81/batchnorm/mul_2MulQfeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_13/batch_normalization_81/batchnorm/mul:z:0*
T0*
_output_shapes
:<2@
>feed_forward_sub_net_13/batch_normalization_81/batchnorm/mul_2�
Ifeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_13_batch_normalization_81_batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype02K
Ifeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_2�
<feed_forward_sub_net_13/batch_normalization_81/batchnorm/subSubQfeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_13/batch_normalization_81/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2>
<feed_forward_sub_net_13/batch_normalization_81/batchnorm/sub�
>feed_forward_sub_net_13/batch_normalization_81/batchnorm/add_1AddV2Bfeed_forward_sub_net_13/batch_normalization_81/batchnorm/mul_1:z:0@feed_forward_sub_net_13/batch_normalization_81/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2@
>feed_forward_sub_net_13/batch_normalization_81/batchnorm/add_1�
feed_forward_sub_net_13/Relu_2ReluBfeed_forward_sub_net_13/batch_normalization_81/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2 
feed_forward_sub_net_13/Relu_2�
6feed_forward_sub_net_13/dense_68/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_13_dense_68_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype028
6feed_forward_sub_net_13/dense_68/MatMul/ReadVariableOp�
'feed_forward_sub_net_13/dense_68/MatMulMatMul,feed_forward_sub_net_13/Relu_2:activations:0>feed_forward_sub_net_13/dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2)
'feed_forward_sub_net_13/dense_68/MatMul�
Gfeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOpReadVariableOpPfeed_forward_sub_net_13_batch_normalization_82_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02I
Gfeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp�
>feed_forward_sub_net_13/batch_normalization_82/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2@
>feed_forward_sub_net_13/batch_normalization_82/batchnorm/add/y�
<feed_forward_sub_net_13/batch_normalization_82/batchnorm/addAddV2Ofeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp:value:0Gfeed_forward_sub_net_13/batch_normalization_82/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2>
<feed_forward_sub_net_13/batch_normalization_82/batchnorm/add�
>feed_forward_sub_net_13/batch_normalization_82/batchnorm/RsqrtRsqrt@feed_forward_sub_net_13/batch_normalization_82/batchnorm/add:z:0*
T0*
_output_shapes
:<2@
>feed_forward_sub_net_13/batch_normalization_82/batchnorm/Rsqrt�
Kfeed_forward_sub_net_13/batch_normalization_82/batchnorm/mul/ReadVariableOpReadVariableOpTfeed_forward_sub_net_13_batch_normalization_82_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02M
Kfeed_forward_sub_net_13/batch_normalization_82/batchnorm/mul/ReadVariableOp�
<feed_forward_sub_net_13/batch_normalization_82/batchnorm/mulMulBfeed_forward_sub_net_13/batch_normalization_82/batchnorm/Rsqrt:y:0Sfeed_forward_sub_net_13/batch_normalization_82/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2>
<feed_forward_sub_net_13/batch_normalization_82/batchnorm/mul�
>feed_forward_sub_net_13/batch_normalization_82/batchnorm/mul_1Mul1feed_forward_sub_net_13/dense_68/MatMul:product:0@feed_forward_sub_net_13/batch_normalization_82/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2@
>feed_forward_sub_net_13/batch_normalization_82/batchnorm/mul_1�
Ifeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_1ReadVariableOpRfeed_forward_sub_net_13_batch_normalization_82_batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype02K
Ifeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_1�
>feed_forward_sub_net_13/batch_normalization_82/batchnorm/mul_2MulQfeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_1:value:0@feed_forward_sub_net_13/batch_normalization_82/batchnorm/mul:z:0*
T0*
_output_shapes
:<2@
>feed_forward_sub_net_13/batch_normalization_82/batchnorm/mul_2�
Ifeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_2ReadVariableOpRfeed_forward_sub_net_13_batch_normalization_82_batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype02K
Ifeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_2�
<feed_forward_sub_net_13/batch_normalization_82/batchnorm/subSubQfeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_2:value:0Bfeed_forward_sub_net_13/batch_normalization_82/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2>
<feed_forward_sub_net_13/batch_normalization_82/batchnorm/sub�
>feed_forward_sub_net_13/batch_normalization_82/batchnorm/add_1AddV2Bfeed_forward_sub_net_13/batch_normalization_82/batchnorm/mul_1:z:0@feed_forward_sub_net_13/batch_normalization_82/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2@
>feed_forward_sub_net_13/batch_normalization_82/batchnorm/add_1�
feed_forward_sub_net_13/Relu_3ReluBfeed_forward_sub_net_13/batch_normalization_82/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2 
feed_forward_sub_net_13/Relu_3�
6feed_forward_sub_net_13/dense_69/MatMul/ReadVariableOpReadVariableOp?feed_forward_sub_net_13_dense_69_matmul_readvariableop_resource*
_output_shapes

:<(*
dtype028
6feed_forward_sub_net_13/dense_69/MatMul/ReadVariableOp�
'feed_forward_sub_net_13/dense_69/MatMulMatMul,feed_forward_sub_net_13/Relu_3:activations:0>feed_forward_sub_net_13/dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2)
'feed_forward_sub_net_13/dense_69/MatMul�
7feed_forward_sub_net_13/dense_69/BiasAdd/ReadVariableOpReadVariableOp@feed_forward_sub_net_13_dense_69_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype029
7feed_forward_sub_net_13/dense_69/BiasAdd/ReadVariableOp�
(feed_forward_sub_net_13/dense_69/BiasAddBiasAdd1feed_forward_sub_net_13/dense_69/MatMul:product:0?feed_forward_sub_net_13/dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2*
(feed_forward_sub_net_13/dense_69/BiasAdd�
IdentityIdentity1feed_forward_sub_net_13/dense_69/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity�
NoOpNoOpH^feed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOpJ^feed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_13/batch_normalization_78/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOpJ^feed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_13/batch_normalization_79/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOpJ^feed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_13/batch_normalization_80/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOpJ^feed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_13/batch_normalization_81/batchnorm/mul/ReadVariableOpH^feed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOpJ^feed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_1J^feed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_2L^feed_forward_sub_net_13/batch_normalization_82/batchnorm/mul/ReadVariableOp7^feed_forward_sub_net_13/dense_65/MatMul/ReadVariableOp7^feed_forward_sub_net_13/dense_66/MatMul/ReadVariableOp7^feed_forward_sub_net_13/dense_67/MatMul/ReadVariableOp7^feed_forward_sub_net_13/dense_68/MatMul/ReadVariableOp8^feed_forward_sub_net_13/dense_69/BiasAdd/ReadVariableOp7^feed_forward_sub_net_13/dense_69/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������(: : : : : : : : : : : : : : : : : : : : : : : : : : 2�
Gfeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOpGfeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp2�
Ifeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_12�
Ifeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_13/batch_normalization_78/batchnorm/ReadVariableOp_22�
Kfeed_forward_sub_net_13/batch_normalization_78/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_13/batch_normalization_78/batchnorm/mul/ReadVariableOp2�
Gfeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOpGfeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp2�
Ifeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_12�
Ifeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_13/batch_normalization_79/batchnorm/ReadVariableOp_22�
Kfeed_forward_sub_net_13/batch_normalization_79/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_13/batch_normalization_79/batchnorm/mul/ReadVariableOp2�
Gfeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOpGfeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp2�
Ifeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_12�
Ifeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_13/batch_normalization_80/batchnorm/ReadVariableOp_22�
Kfeed_forward_sub_net_13/batch_normalization_80/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_13/batch_normalization_80/batchnorm/mul/ReadVariableOp2�
Gfeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOpGfeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp2�
Ifeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_12�
Ifeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_13/batch_normalization_81/batchnorm/ReadVariableOp_22�
Kfeed_forward_sub_net_13/batch_normalization_81/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_13/batch_normalization_81/batchnorm/mul/ReadVariableOp2�
Gfeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOpGfeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp2�
Ifeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_1Ifeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_12�
Ifeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_2Ifeed_forward_sub_net_13/batch_normalization_82/batchnorm/ReadVariableOp_22�
Kfeed_forward_sub_net_13/batch_normalization_82/batchnorm/mul/ReadVariableOpKfeed_forward_sub_net_13/batch_normalization_82/batchnorm/mul/ReadVariableOp2p
6feed_forward_sub_net_13/dense_65/MatMul/ReadVariableOp6feed_forward_sub_net_13/dense_65/MatMul/ReadVariableOp2p
6feed_forward_sub_net_13/dense_66/MatMul/ReadVariableOp6feed_forward_sub_net_13/dense_66/MatMul/ReadVariableOp2p
6feed_forward_sub_net_13/dense_67/MatMul/ReadVariableOp6feed_forward_sub_net_13/dense_67/MatMul/ReadVariableOp2p
6feed_forward_sub_net_13/dense_68/MatMul/ReadVariableOp6feed_forward_sub_net_13/dense_68/MatMul/ReadVariableOp2r
7feed_forward_sub_net_13/dense_69/BiasAdd/ReadVariableOp7feed_forward_sub_net_13/dense_69/BiasAdd/ReadVariableOp2p
6feed_forward_sub_net_13/dense_69/MatMul/ReadVariableOp6feed_forward_sub_net_13/dense_69/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������(
!
_user_specified_name	input_1
�
�
E__inference_dense_66_layer_call_and_return_conditional_losses_7059985

inputs0
matmul_readvariableop_resource:<<
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������<: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�E
�
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7058431
x,
batch_normalization_78_7058364:(,
batch_normalization_78_7058366:(,
batch_normalization_78_7058368:(,
batch_normalization_78_7058370:("
dense_65_7058373:(<,
batch_normalization_79_7058376:<,
batch_normalization_79_7058378:<,
batch_normalization_79_7058380:<,
batch_normalization_79_7058382:<"
dense_66_7058386:<<,
batch_normalization_80_7058389:<,
batch_normalization_80_7058391:<,
batch_normalization_80_7058393:<,
batch_normalization_80_7058395:<"
dense_67_7058399:<<,
batch_normalization_81_7058402:<,
batch_normalization_81_7058404:<,
batch_normalization_81_7058406:<,
batch_normalization_81_7058408:<"
dense_68_7058412:<<,
batch_normalization_82_7058415:<,
batch_normalization_82_7058417:<,
batch_normalization_82_7058419:<,
batch_normalization_82_7058421:<"
dense_69_7058425:<(
dense_69_7058427:(
identity��.batch_normalization_78/StatefulPartitionedCall�.batch_normalization_79/StatefulPartitionedCall�.batch_normalization_80/StatefulPartitionedCall�.batch_normalization_81/StatefulPartitionedCall�.batch_normalization_82/StatefulPartitionedCall� dense_65/StatefulPartitionedCall� dense_66/StatefulPartitionedCall� dense_67/StatefulPartitionedCall� dense_68/StatefulPartitionedCall� dense_69/StatefulPartitionedCall�
.batch_normalization_78/StatefulPartitionedCallStatefulPartitionedCallxbatch_normalization_78_7058364batch_normalization_78_7058366batch_normalization_78_7058368batch_normalization_78_7058370*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_78_layer_call_and_return_conditional_losses_705734420
.batch_normalization_78/StatefulPartitionedCall�
 dense_65/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_78/StatefulPartitionedCall:output:0dense_65_7058373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_70581112"
 dense_65/StatefulPartitionedCall�
.batch_normalization_79/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0batch_normalization_79_7058376batch_normalization_79_7058378batch_normalization_79_7058380batch_normalization_79_7058382*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_79_layer_call_and_return_conditional_losses_705751020
.batch_normalization_79/StatefulPartitionedCall
ReluRelu7batch_normalization_79/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������<2
Relu�
 dense_66/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0dense_66_7058386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_66_layer_call_and_return_conditional_losses_70581322"
 dense_66/StatefulPartitionedCall�
.batch_normalization_80/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0batch_normalization_80_7058389batch_normalization_80_7058391batch_normalization_80_7058393batch_normalization_80_7058395*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_80_layer_call_and_return_conditional_losses_705767620
.batch_normalization_80/StatefulPartitionedCall�
Relu_1Relu7batch_normalization_80/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������<2
Relu_1�
 dense_67/StatefulPartitionedCallStatefulPartitionedCallRelu_1:activations:0dense_67_7058399*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_67_layer_call_and_return_conditional_losses_70581532"
 dense_67/StatefulPartitionedCall�
.batch_normalization_81/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0batch_normalization_81_7058402batch_normalization_81_7058404batch_normalization_81_7058406batch_normalization_81_7058408*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_81_layer_call_and_return_conditional_losses_705784220
.batch_normalization_81/StatefulPartitionedCall�
Relu_2Relu7batch_normalization_81/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������<2
Relu_2�
 dense_68/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_68_7058412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_68_layer_call_and_return_conditional_losses_70581742"
 dense_68/StatefulPartitionedCall�
.batch_normalization_82/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0batch_normalization_82_7058415batch_normalization_82_7058417batch_normalization_82_7058419batch_normalization_82_7058421*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_82_layer_call_and_return_conditional_losses_705800820
.batch_normalization_82/StatefulPartitionedCall�
Relu_3Relu7batch_normalization_82/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������<2
Relu_3�
 dense_69/StatefulPartitionedCallStatefulPartitionedCallRelu_3:activations:0dense_69_7058425dense_69_7058427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_69_layer_call_and_return_conditional_losses_70581982"
 dense_69/StatefulPartitionedCall�
IdentityIdentity)dense_69/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity�
NoOpNoOp/^batch_normalization_78/StatefulPartitionedCall/^batch_normalization_79/StatefulPartitionedCall/^batch_normalization_80/StatefulPartitionedCall/^batch_normalization_81/StatefulPartitionedCall/^batch_normalization_82/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������(: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_78/StatefulPartitionedCall.batch_normalization_78/StatefulPartitionedCall2`
.batch_normalization_79/StatefulPartitionedCall.batch_normalization_79/StatefulPartitionedCall2`
.batch_normalization_80/StatefulPartitionedCall.batch_normalization_80/StatefulPartitionedCall2`
.batch_normalization_81/StatefulPartitionedCall.batch_normalization_81/StatefulPartitionedCall2`
.batch_normalization_82/StatefulPartitionedCall.batch_normalization_82/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall:J F
'
_output_shapes
:���������(

_user_specified_namex
�
�
9__inference_feed_forward_sub_net_13_layer_call_fn_7059554
input_1
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(<
	unknown_4:<
	unknown_5:<
	unknown_6:<
	unknown_7:<
	unknown_8:<<
	unknown_9:<

unknown_10:<

unknown_11:<

unknown_12:<

unknown_13:<<

unknown_14:<

unknown_15:<

unknown_16:<

unknown_17:<

unknown_18:<<

unknown_19:<

unknown_20:<

unknown_21:<

unknown_22:<

unknown_23:<(

unknown_24:(
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
:���������(*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_70584312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������(: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������(
!
_user_specified_name	input_1
�
�
E__inference_dense_68_layer_call_and_return_conditional_losses_7058174

inputs0
matmul_readvariableop_resource:<<
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������<: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_81_layer_call_and_return_conditional_losses_7057780

inputs/
!batchnorm_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:<1
#batchnorm_readvariableop_1_resource:<1
#batchnorm_readvariableop_2_resource:<
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
��
�
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7059034
xL
>batch_normalization_78_assignmovingavg_readvariableop_resource:(N
@batch_normalization_78_assignmovingavg_1_readvariableop_resource:(J
<batch_normalization_78_batchnorm_mul_readvariableop_resource:(F
8batch_normalization_78_batchnorm_readvariableop_resource:(9
'dense_65_matmul_readvariableop_resource:(<L
>batch_normalization_79_assignmovingavg_readvariableop_resource:<N
@batch_normalization_79_assignmovingavg_1_readvariableop_resource:<J
<batch_normalization_79_batchnorm_mul_readvariableop_resource:<F
8batch_normalization_79_batchnorm_readvariableop_resource:<9
'dense_66_matmul_readvariableop_resource:<<L
>batch_normalization_80_assignmovingavg_readvariableop_resource:<N
@batch_normalization_80_assignmovingavg_1_readvariableop_resource:<J
<batch_normalization_80_batchnorm_mul_readvariableop_resource:<F
8batch_normalization_80_batchnorm_readvariableop_resource:<9
'dense_67_matmul_readvariableop_resource:<<L
>batch_normalization_81_assignmovingavg_readvariableop_resource:<N
@batch_normalization_81_assignmovingavg_1_readvariableop_resource:<J
<batch_normalization_81_batchnorm_mul_readvariableop_resource:<F
8batch_normalization_81_batchnorm_readvariableop_resource:<9
'dense_68_matmul_readvariableop_resource:<<L
>batch_normalization_82_assignmovingavg_readvariableop_resource:<N
@batch_normalization_82_assignmovingavg_1_readvariableop_resource:<J
<batch_normalization_82_batchnorm_mul_readvariableop_resource:<F
8batch_normalization_82_batchnorm_readvariableop_resource:<9
'dense_69_matmul_readvariableop_resource:<(6
(dense_69_biasadd_readvariableop_resource:(
identity��&batch_normalization_78/AssignMovingAvg�5batch_normalization_78/AssignMovingAvg/ReadVariableOp�(batch_normalization_78/AssignMovingAvg_1�7batch_normalization_78/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_78/batchnorm/ReadVariableOp�3batch_normalization_78/batchnorm/mul/ReadVariableOp�&batch_normalization_79/AssignMovingAvg�5batch_normalization_79/AssignMovingAvg/ReadVariableOp�(batch_normalization_79/AssignMovingAvg_1�7batch_normalization_79/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_79/batchnorm/ReadVariableOp�3batch_normalization_79/batchnorm/mul/ReadVariableOp�&batch_normalization_80/AssignMovingAvg�5batch_normalization_80/AssignMovingAvg/ReadVariableOp�(batch_normalization_80/AssignMovingAvg_1�7batch_normalization_80/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_80/batchnorm/ReadVariableOp�3batch_normalization_80/batchnorm/mul/ReadVariableOp�&batch_normalization_81/AssignMovingAvg�5batch_normalization_81/AssignMovingAvg/ReadVariableOp�(batch_normalization_81/AssignMovingAvg_1�7batch_normalization_81/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_81/batchnorm/ReadVariableOp�3batch_normalization_81/batchnorm/mul/ReadVariableOp�&batch_normalization_82/AssignMovingAvg�5batch_normalization_82/AssignMovingAvg/ReadVariableOp�(batch_normalization_82/AssignMovingAvg_1�7batch_normalization_82/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_82/batchnorm/ReadVariableOp�3batch_normalization_82/batchnorm/mul/ReadVariableOp�dense_65/MatMul/ReadVariableOp�dense_66/MatMul/ReadVariableOp�dense_67/MatMul/ReadVariableOp�dense_68/MatMul/ReadVariableOp�dense_69/BiasAdd/ReadVariableOp�dense_69/MatMul/ReadVariableOp�
5batch_normalization_78/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_78/moments/mean/reduction_indices�
#batch_normalization_78/moments/meanMeanx>batch_normalization_78/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:(*
	keep_dims(2%
#batch_normalization_78/moments/mean�
+batch_normalization_78/moments/StopGradientStopGradient,batch_normalization_78/moments/mean:output:0*
T0*
_output_shapes

:(2-
+batch_normalization_78/moments/StopGradient�
0batch_normalization_78/moments/SquaredDifferenceSquaredDifferencex4batch_normalization_78/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������(22
0batch_normalization_78/moments/SquaredDifference�
9batch_normalization_78/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_78/moments/variance/reduction_indices�
'batch_normalization_78/moments/varianceMean4batch_normalization_78/moments/SquaredDifference:z:0Bbatch_normalization_78/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:(*
	keep_dims(2)
'batch_normalization_78/moments/variance�
&batch_normalization_78/moments/SqueezeSqueeze,batch_normalization_78/moments/mean:output:0*
T0*
_output_shapes
:(*
squeeze_dims
 2(
&batch_normalization_78/moments/Squeeze�
(batch_normalization_78/moments/Squeeze_1Squeeze0batch_normalization_78/moments/variance:output:0*
T0*
_output_shapes
:(*
squeeze_dims
 2*
(batch_normalization_78/moments/Squeeze_1�
,batch_normalization_78/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_78/AssignMovingAvg/decay�
+batch_normalization_78/AssignMovingAvg/CastCast5batch_normalization_78/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_78/AssignMovingAvg/Cast�
5batch_normalization_78/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_78_assignmovingavg_readvariableop_resource*
_output_shapes
:(*
dtype027
5batch_normalization_78/AssignMovingAvg/ReadVariableOp�
*batch_normalization_78/AssignMovingAvg/subSub=batch_normalization_78/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_78/moments/Squeeze:output:0*
T0*
_output_shapes
:(2,
*batch_normalization_78/AssignMovingAvg/sub�
*batch_normalization_78/AssignMovingAvg/mulMul.batch_normalization_78/AssignMovingAvg/sub:z:0/batch_normalization_78/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:(2,
*batch_normalization_78/AssignMovingAvg/mul�
&batch_normalization_78/AssignMovingAvgAssignSubVariableOp>batch_normalization_78_assignmovingavg_readvariableop_resource.batch_normalization_78/AssignMovingAvg/mul:z:06^batch_normalization_78/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_78/AssignMovingAvg�
.batch_normalization_78/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_78/AssignMovingAvg_1/decay�
-batch_normalization_78/AssignMovingAvg_1/CastCast7batch_normalization_78/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_78/AssignMovingAvg_1/Cast�
7batch_normalization_78/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_78_assignmovingavg_1_readvariableop_resource*
_output_shapes
:(*
dtype029
7batch_normalization_78/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_78/AssignMovingAvg_1/subSub?batch_normalization_78/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_78/moments/Squeeze_1:output:0*
T0*
_output_shapes
:(2.
,batch_normalization_78/AssignMovingAvg_1/sub�
,batch_normalization_78/AssignMovingAvg_1/mulMul0batch_normalization_78/AssignMovingAvg_1/sub:z:01batch_normalization_78/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:(2.
,batch_normalization_78/AssignMovingAvg_1/mul�
(batch_normalization_78/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_78_assignmovingavg_1_readvariableop_resource0batch_normalization_78/AssignMovingAvg_1/mul:z:08^batch_normalization_78/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_78/AssignMovingAvg_1�
&batch_normalization_78/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_78/batchnorm/add/y�
$batch_normalization_78/batchnorm/addAddV21batch_normalization_78/moments/Squeeze_1:output:0/batch_normalization_78/batchnorm/add/y:output:0*
T0*
_output_shapes
:(2&
$batch_normalization_78/batchnorm/add�
&batch_normalization_78/batchnorm/RsqrtRsqrt(batch_normalization_78/batchnorm/add:z:0*
T0*
_output_shapes
:(2(
&batch_normalization_78/batchnorm/Rsqrt�
3batch_normalization_78/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_78_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype025
3batch_normalization_78/batchnorm/mul/ReadVariableOp�
$batch_normalization_78/batchnorm/mulMul*batch_normalization_78/batchnorm/Rsqrt:y:0;batch_normalization_78/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2&
$batch_normalization_78/batchnorm/mul�
&batch_normalization_78/batchnorm/mul_1Mulx(batch_normalization_78/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2(
&batch_normalization_78/batchnorm/mul_1�
&batch_normalization_78/batchnorm/mul_2Mul/batch_normalization_78/moments/Squeeze:output:0(batch_normalization_78/batchnorm/mul:z:0*
T0*
_output_shapes
:(2(
&batch_normalization_78/batchnorm/mul_2�
/batch_normalization_78/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_78_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype021
/batch_normalization_78/batchnorm/ReadVariableOp�
$batch_normalization_78/batchnorm/subSub7batch_normalization_78/batchnorm/ReadVariableOp:value:0*batch_normalization_78/batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2&
$batch_normalization_78/batchnorm/sub�
&batch_normalization_78/batchnorm/add_1AddV2*batch_normalization_78/batchnorm/mul_1:z:0(batch_normalization_78/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2(
&batch_normalization_78/batchnorm/add_1�
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02 
dense_65/MatMul/ReadVariableOp�
dense_65/MatMulMatMul*batch_normalization_78/batchnorm/add_1:z:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_65/MatMul�
5batch_normalization_79/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_79/moments/mean/reduction_indices�
#batch_normalization_79/moments/meanMeandense_65/MatMul:product:0>batch_normalization_79/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2%
#batch_normalization_79/moments/mean�
+batch_normalization_79/moments/StopGradientStopGradient,batch_normalization_79/moments/mean:output:0*
T0*
_output_shapes

:<2-
+batch_normalization_79/moments/StopGradient�
0batch_normalization_79/moments/SquaredDifferenceSquaredDifferencedense_65/MatMul:product:04batch_normalization_79/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������<22
0batch_normalization_79/moments/SquaredDifference�
9batch_normalization_79/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_79/moments/variance/reduction_indices�
'batch_normalization_79/moments/varianceMean4batch_normalization_79/moments/SquaredDifference:z:0Bbatch_normalization_79/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2)
'batch_normalization_79/moments/variance�
&batch_normalization_79/moments/SqueezeSqueeze,batch_normalization_79/moments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2(
&batch_normalization_79/moments/Squeeze�
(batch_normalization_79/moments/Squeeze_1Squeeze0batch_normalization_79/moments/variance:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2*
(batch_normalization_79/moments/Squeeze_1�
,batch_normalization_79/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_79/AssignMovingAvg/decay�
+batch_normalization_79/AssignMovingAvg/CastCast5batch_normalization_79/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_79/AssignMovingAvg/Cast�
5batch_normalization_79/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_79_assignmovingavg_readvariableop_resource*
_output_shapes
:<*
dtype027
5batch_normalization_79/AssignMovingAvg/ReadVariableOp�
*batch_normalization_79/AssignMovingAvg/subSub=batch_normalization_79/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_79/moments/Squeeze:output:0*
T0*
_output_shapes
:<2,
*batch_normalization_79/AssignMovingAvg/sub�
*batch_normalization_79/AssignMovingAvg/mulMul.batch_normalization_79/AssignMovingAvg/sub:z:0/batch_normalization_79/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2,
*batch_normalization_79/AssignMovingAvg/mul�
&batch_normalization_79/AssignMovingAvgAssignSubVariableOp>batch_normalization_79_assignmovingavg_readvariableop_resource.batch_normalization_79/AssignMovingAvg/mul:z:06^batch_normalization_79/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_79/AssignMovingAvg�
.batch_normalization_79/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_79/AssignMovingAvg_1/decay�
-batch_normalization_79/AssignMovingAvg_1/CastCast7batch_normalization_79/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_79/AssignMovingAvg_1/Cast�
7batch_normalization_79/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_79_assignmovingavg_1_readvariableop_resource*
_output_shapes
:<*
dtype029
7batch_normalization_79/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_79/AssignMovingAvg_1/subSub?batch_normalization_79/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_79/moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2.
,batch_normalization_79/AssignMovingAvg_1/sub�
,batch_normalization_79/AssignMovingAvg_1/mulMul0batch_normalization_79/AssignMovingAvg_1/sub:z:01batch_normalization_79/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2.
,batch_normalization_79/AssignMovingAvg_1/mul�
(batch_normalization_79/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_79_assignmovingavg_1_readvariableop_resource0batch_normalization_79/AssignMovingAvg_1/mul:z:08^batch_normalization_79/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_79/AssignMovingAvg_1�
&batch_normalization_79/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_79/batchnorm/add/y�
$batch_normalization_79/batchnorm/addAddV21batch_normalization_79/moments/Squeeze_1:output:0/batch_normalization_79/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_79/batchnorm/add�
&batch_normalization_79/batchnorm/RsqrtRsqrt(batch_normalization_79/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_79/batchnorm/Rsqrt�
3batch_normalization_79/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_79_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_79/batchnorm/mul/ReadVariableOp�
$batch_normalization_79/batchnorm/mulMul*batch_normalization_79/batchnorm/Rsqrt:y:0;batch_normalization_79/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_79/batchnorm/mul�
&batch_normalization_79/batchnorm/mul_1Muldense_65/MatMul:product:0(batch_normalization_79/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_79/batchnorm/mul_1�
&batch_normalization_79/batchnorm/mul_2Mul/batch_normalization_79/moments/Squeeze:output:0(batch_normalization_79/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_79/batchnorm/mul_2�
/batch_normalization_79/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_79_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_79/batchnorm/ReadVariableOp�
$batch_normalization_79/batchnorm/subSub7batch_normalization_79/batchnorm/ReadVariableOp:value:0*batch_normalization_79/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_79/batchnorm/sub�
&batch_normalization_79/batchnorm/add_1AddV2*batch_normalization_79/batchnorm/mul_1:z:0(batch_normalization_79/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_79/batchnorm/add_1r
ReluRelu*batch_normalization_79/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu�
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02 
dense_66/MatMul/ReadVariableOp�
dense_66/MatMulMatMulRelu:activations:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_66/MatMul�
5batch_normalization_80/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_80/moments/mean/reduction_indices�
#batch_normalization_80/moments/meanMeandense_66/MatMul:product:0>batch_normalization_80/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2%
#batch_normalization_80/moments/mean�
+batch_normalization_80/moments/StopGradientStopGradient,batch_normalization_80/moments/mean:output:0*
T0*
_output_shapes

:<2-
+batch_normalization_80/moments/StopGradient�
0batch_normalization_80/moments/SquaredDifferenceSquaredDifferencedense_66/MatMul:product:04batch_normalization_80/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������<22
0batch_normalization_80/moments/SquaredDifference�
9batch_normalization_80/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_80/moments/variance/reduction_indices�
'batch_normalization_80/moments/varianceMean4batch_normalization_80/moments/SquaredDifference:z:0Bbatch_normalization_80/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2)
'batch_normalization_80/moments/variance�
&batch_normalization_80/moments/SqueezeSqueeze,batch_normalization_80/moments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2(
&batch_normalization_80/moments/Squeeze�
(batch_normalization_80/moments/Squeeze_1Squeeze0batch_normalization_80/moments/variance:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2*
(batch_normalization_80/moments/Squeeze_1�
,batch_normalization_80/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_80/AssignMovingAvg/decay�
+batch_normalization_80/AssignMovingAvg/CastCast5batch_normalization_80/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_80/AssignMovingAvg/Cast�
5batch_normalization_80/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_80_assignmovingavg_readvariableop_resource*
_output_shapes
:<*
dtype027
5batch_normalization_80/AssignMovingAvg/ReadVariableOp�
*batch_normalization_80/AssignMovingAvg/subSub=batch_normalization_80/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_80/moments/Squeeze:output:0*
T0*
_output_shapes
:<2,
*batch_normalization_80/AssignMovingAvg/sub�
*batch_normalization_80/AssignMovingAvg/mulMul.batch_normalization_80/AssignMovingAvg/sub:z:0/batch_normalization_80/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2,
*batch_normalization_80/AssignMovingAvg/mul�
&batch_normalization_80/AssignMovingAvgAssignSubVariableOp>batch_normalization_80_assignmovingavg_readvariableop_resource.batch_normalization_80/AssignMovingAvg/mul:z:06^batch_normalization_80/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_80/AssignMovingAvg�
.batch_normalization_80/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_80/AssignMovingAvg_1/decay�
-batch_normalization_80/AssignMovingAvg_1/CastCast7batch_normalization_80/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_80/AssignMovingAvg_1/Cast�
7batch_normalization_80/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_80_assignmovingavg_1_readvariableop_resource*
_output_shapes
:<*
dtype029
7batch_normalization_80/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_80/AssignMovingAvg_1/subSub?batch_normalization_80/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_80/moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2.
,batch_normalization_80/AssignMovingAvg_1/sub�
,batch_normalization_80/AssignMovingAvg_1/mulMul0batch_normalization_80/AssignMovingAvg_1/sub:z:01batch_normalization_80/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2.
,batch_normalization_80/AssignMovingAvg_1/mul�
(batch_normalization_80/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_80_assignmovingavg_1_readvariableop_resource0batch_normalization_80/AssignMovingAvg_1/mul:z:08^batch_normalization_80/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_80/AssignMovingAvg_1�
&batch_normalization_80/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_80/batchnorm/add/y�
$batch_normalization_80/batchnorm/addAddV21batch_normalization_80/moments/Squeeze_1:output:0/batch_normalization_80/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_80/batchnorm/add�
&batch_normalization_80/batchnorm/RsqrtRsqrt(batch_normalization_80/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_80/batchnorm/Rsqrt�
3batch_normalization_80/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_80_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_80/batchnorm/mul/ReadVariableOp�
$batch_normalization_80/batchnorm/mulMul*batch_normalization_80/batchnorm/Rsqrt:y:0;batch_normalization_80/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_80/batchnorm/mul�
&batch_normalization_80/batchnorm/mul_1Muldense_66/MatMul:product:0(batch_normalization_80/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_80/batchnorm/mul_1�
&batch_normalization_80/batchnorm/mul_2Mul/batch_normalization_80/moments/Squeeze:output:0(batch_normalization_80/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_80/batchnorm/mul_2�
/batch_normalization_80/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_80_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_80/batchnorm/ReadVariableOp�
$batch_normalization_80/batchnorm/subSub7batch_normalization_80/batchnorm/ReadVariableOp:value:0*batch_normalization_80/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_80/batchnorm/sub�
&batch_normalization_80/batchnorm/add_1AddV2*batch_normalization_80/batchnorm/mul_1:z:0(batch_normalization_80/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_80/batchnorm/add_1v
Relu_1Relu*batch_normalization_80/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu_1�
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02 
dense_67/MatMul/ReadVariableOp�
dense_67/MatMulMatMulRelu_1:activations:0&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_67/MatMul�
5batch_normalization_81/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_81/moments/mean/reduction_indices�
#batch_normalization_81/moments/meanMeandense_67/MatMul:product:0>batch_normalization_81/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2%
#batch_normalization_81/moments/mean�
+batch_normalization_81/moments/StopGradientStopGradient,batch_normalization_81/moments/mean:output:0*
T0*
_output_shapes

:<2-
+batch_normalization_81/moments/StopGradient�
0batch_normalization_81/moments/SquaredDifferenceSquaredDifferencedense_67/MatMul:product:04batch_normalization_81/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������<22
0batch_normalization_81/moments/SquaredDifference�
9batch_normalization_81/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_81/moments/variance/reduction_indices�
'batch_normalization_81/moments/varianceMean4batch_normalization_81/moments/SquaredDifference:z:0Bbatch_normalization_81/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2)
'batch_normalization_81/moments/variance�
&batch_normalization_81/moments/SqueezeSqueeze,batch_normalization_81/moments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2(
&batch_normalization_81/moments/Squeeze�
(batch_normalization_81/moments/Squeeze_1Squeeze0batch_normalization_81/moments/variance:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2*
(batch_normalization_81/moments/Squeeze_1�
,batch_normalization_81/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_81/AssignMovingAvg/decay�
+batch_normalization_81/AssignMovingAvg/CastCast5batch_normalization_81/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_81/AssignMovingAvg/Cast�
5batch_normalization_81/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_81_assignmovingavg_readvariableop_resource*
_output_shapes
:<*
dtype027
5batch_normalization_81/AssignMovingAvg/ReadVariableOp�
*batch_normalization_81/AssignMovingAvg/subSub=batch_normalization_81/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_81/moments/Squeeze:output:0*
T0*
_output_shapes
:<2,
*batch_normalization_81/AssignMovingAvg/sub�
*batch_normalization_81/AssignMovingAvg/mulMul.batch_normalization_81/AssignMovingAvg/sub:z:0/batch_normalization_81/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2,
*batch_normalization_81/AssignMovingAvg/mul�
&batch_normalization_81/AssignMovingAvgAssignSubVariableOp>batch_normalization_81_assignmovingavg_readvariableop_resource.batch_normalization_81/AssignMovingAvg/mul:z:06^batch_normalization_81/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_81/AssignMovingAvg�
.batch_normalization_81/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_81/AssignMovingAvg_1/decay�
-batch_normalization_81/AssignMovingAvg_1/CastCast7batch_normalization_81/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_81/AssignMovingAvg_1/Cast�
7batch_normalization_81/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_81_assignmovingavg_1_readvariableop_resource*
_output_shapes
:<*
dtype029
7batch_normalization_81/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_81/AssignMovingAvg_1/subSub?batch_normalization_81/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_81/moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2.
,batch_normalization_81/AssignMovingAvg_1/sub�
,batch_normalization_81/AssignMovingAvg_1/mulMul0batch_normalization_81/AssignMovingAvg_1/sub:z:01batch_normalization_81/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2.
,batch_normalization_81/AssignMovingAvg_1/mul�
(batch_normalization_81/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_81_assignmovingavg_1_readvariableop_resource0batch_normalization_81/AssignMovingAvg_1/mul:z:08^batch_normalization_81/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_81/AssignMovingAvg_1�
&batch_normalization_81/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_81/batchnorm/add/y�
$batch_normalization_81/batchnorm/addAddV21batch_normalization_81/moments/Squeeze_1:output:0/batch_normalization_81/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_81/batchnorm/add�
&batch_normalization_81/batchnorm/RsqrtRsqrt(batch_normalization_81/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_81/batchnorm/Rsqrt�
3batch_normalization_81/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_81_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_81/batchnorm/mul/ReadVariableOp�
$batch_normalization_81/batchnorm/mulMul*batch_normalization_81/batchnorm/Rsqrt:y:0;batch_normalization_81/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_81/batchnorm/mul�
&batch_normalization_81/batchnorm/mul_1Muldense_67/MatMul:product:0(batch_normalization_81/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_81/batchnorm/mul_1�
&batch_normalization_81/batchnorm/mul_2Mul/batch_normalization_81/moments/Squeeze:output:0(batch_normalization_81/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_81/batchnorm/mul_2�
/batch_normalization_81/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_81_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_81/batchnorm/ReadVariableOp�
$batch_normalization_81/batchnorm/subSub7batch_normalization_81/batchnorm/ReadVariableOp:value:0*batch_normalization_81/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_81/batchnorm/sub�
&batch_normalization_81/batchnorm/add_1AddV2*batch_normalization_81/batchnorm/mul_1:z:0(batch_normalization_81/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_81/batchnorm/add_1v
Relu_2Relu*batch_normalization_81/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu_2�
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02 
dense_68/MatMul/ReadVariableOp�
dense_68/MatMulMatMulRelu_2:activations:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_68/MatMul�
5batch_normalization_82/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_82/moments/mean/reduction_indices�
#batch_normalization_82/moments/meanMeandense_68/MatMul:product:0>batch_normalization_82/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2%
#batch_normalization_82/moments/mean�
+batch_normalization_82/moments/StopGradientStopGradient,batch_normalization_82/moments/mean:output:0*
T0*
_output_shapes

:<2-
+batch_normalization_82/moments/StopGradient�
0batch_normalization_82/moments/SquaredDifferenceSquaredDifferencedense_68/MatMul:product:04batch_normalization_82/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������<22
0batch_normalization_82/moments/SquaredDifference�
9batch_normalization_82/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_82/moments/variance/reduction_indices�
'batch_normalization_82/moments/varianceMean4batch_normalization_82/moments/SquaredDifference:z:0Bbatch_normalization_82/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2)
'batch_normalization_82/moments/variance�
&batch_normalization_82/moments/SqueezeSqueeze,batch_normalization_82/moments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2(
&batch_normalization_82/moments/Squeeze�
(batch_normalization_82/moments/Squeeze_1Squeeze0batch_normalization_82/moments/variance:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2*
(batch_normalization_82/moments/Squeeze_1�
,batch_normalization_82/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_82/AssignMovingAvg/decay�
+batch_normalization_82/AssignMovingAvg/CastCast5batch_normalization_82/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_82/AssignMovingAvg/Cast�
5batch_normalization_82/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_82_assignmovingavg_readvariableop_resource*
_output_shapes
:<*
dtype027
5batch_normalization_82/AssignMovingAvg/ReadVariableOp�
*batch_normalization_82/AssignMovingAvg/subSub=batch_normalization_82/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_82/moments/Squeeze:output:0*
T0*
_output_shapes
:<2,
*batch_normalization_82/AssignMovingAvg/sub�
*batch_normalization_82/AssignMovingAvg/mulMul.batch_normalization_82/AssignMovingAvg/sub:z:0/batch_normalization_82/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2,
*batch_normalization_82/AssignMovingAvg/mul�
&batch_normalization_82/AssignMovingAvgAssignSubVariableOp>batch_normalization_82_assignmovingavg_readvariableop_resource.batch_normalization_82/AssignMovingAvg/mul:z:06^batch_normalization_82/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_82/AssignMovingAvg�
.batch_normalization_82/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_82/AssignMovingAvg_1/decay�
-batch_normalization_82/AssignMovingAvg_1/CastCast7batch_normalization_82/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_82/AssignMovingAvg_1/Cast�
7batch_normalization_82/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_82_assignmovingavg_1_readvariableop_resource*
_output_shapes
:<*
dtype029
7batch_normalization_82/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_82/AssignMovingAvg_1/subSub?batch_normalization_82/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_82/moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2.
,batch_normalization_82/AssignMovingAvg_1/sub�
,batch_normalization_82/AssignMovingAvg_1/mulMul0batch_normalization_82/AssignMovingAvg_1/sub:z:01batch_normalization_82/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2.
,batch_normalization_82/AssignMovingAvg_1/mul�
(batch_normalization_82/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_82_assignmovingavg_1_readvariableop_resource0batch_normalization_82/AssignMovingAvg_1/mul:z:08^batch_normalization_82/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_82/AssignMovingAvg_1�
&batch_normalization_82/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_82/batchnorm/add/y�
$batch_normalization_82/batchnorm/addAddV21batch_normalization_82/moments/Squeeze_1:output:0/batch_normalization_82/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_82/batchnorm/add�
&batch_normalization_82/batchnorm/RsqrtRsqrt(batch_normalization_82/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_82/batchnorm/Rsqrt�
3batch_normalization_82/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_82_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_82/batchnorm/mul/ReadVariableOp�
$batch_normalization_82/batchnorm/mulMul*batch_normalization_82/batchnorm/Rsqrt:y:0;batch_normalization_82/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_82/batchnorm/mul�
&batch_normalization_82/batchnorm/mul_1Muldense_68/MatMul:product:0(batch_normalization_82/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_82/batchnorm/mul_1�
&batch_normalization_82/batchnorm/mul_2Mul/batch_normalization_82/moments/Squeeze:output:0(batch_normalization_82/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_82/batchnorm/mul_2�
/batch_normalization_82/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_82_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_82/batchnorm/ReadVariableOp�
$batch_normalization_82/batchnorm/subSub7batch_normalization_82/batchnorm/ReadVariableOp:value:0*batch_normalization_82/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_82/batchnorm/sub�
&batch_normalization_82/batchnorm/add_1AddV2*batch_normalization_82/batchnorm/mul_1:z:0(batch_normalization_82/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_82/batchnorm/add_1v
Relu_3Relu*batch_normalization_82/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu_3�
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource*
_output_shapes

:<(*
dtype02 
dense_69/MatMul/ReadVariableOp�
dense_69/MatMulMatMulRelu_3:activations:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_69/MatMul�
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02!
dense_69/BiasAdd/ReadVariableOp�
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_69/BiasAddt
IdentityIdentitydense_69/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity�
NoOpNoOp'^batch_normalization_78/AssignMovingAvg6^batch_normalization_78/AssignMovingAvg/ReadVariableOp)^batch_normalization_78/AssignMovingAvg_18^batch_normalization_78/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_78/batchnorm/ReadVariableOp4^batch_normalization_78/batchnorm/mul/ReadVariableOp'^batch_normalization_79/AssignMovingAvg6^batch_normalization_79/AssignMovingAvg/ReadVariableOp)^batch_normalization_79/AssignMovingAvg_18^batch_normalization_79/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_79/batchnorm/ReadVariableOp4^batch_normalization_79/batchnorm/mul/ReadVariableOp'^batch_normalization_80/AssignMovingAvg6^batch_normalization_80/AssignMovingAvg/ReadVariableOp)^batch_normalization_80/AssignMovingAvg_18^batch_normalization_80/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_80/batchnorm/ReadVariableOp4^batch_normalization_80/batchnorm/mul/ReadVariableOp'^batch_normalization_81/AssignMovingAvg6^batch_normalization_81/AssignMovingAvg/ReadVariableOp)^batch_normalization_81/AssignMovingAvg_18^batch_normalization_81/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_81/batchnorm/ReadVariableOp4^batch_normalization_81/batchnorm/mul/ReadVariableOp'^batch_normalization_82/AssignMovingAvg6^batch_normalization_82/AssignMovingAvg/ReadVariableOp)^batch_normalization_82/AssignMovingAvg_18^batch_normalization_82/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_82/batchnorm/ReadVariableOp4^batch_normalization_82/batchnorm/mul/ReadVariableOp^dense_65/MatMul/ReadVariableOp^dense_66/MatMul/ReadVariableOp^dense_67/MatMul/ReadVariableOp^dense_68/MatMul/ReadVariableOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������(: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_78/AssignMovingAvg&batch_normalization_78/AssignMovingAvg2n
5batch_normalization_78/AssignMovingAvg/ReadVariableOp5batch_normalization_78/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_78/AssignMovingAvg_1(batch_normalization_78/AssignMovingAvg_12r
7batch_normalization_78/AssignMovingAvg_1/ReadVariableOp7batch_normalization_78/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_78/batchnorm/ReadVariableOp/batch_normalization_78/batchnorm/ReadVariableOp2j
3batch_normalization_78/batchnorm/mul/ReadVariableOp3batch_normalization_78/batchnorm/mul/ReadVariableOp2P
&batch_normalization_79/AssignMovingAvg&batch_normalization_79/AssignMovingAvg2n
5batch_normalization_79/AssignMovingAvg/ReadVariableOp5batch_normalization_79/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_79/AssignMovingAvg_1(batch_normalization_79/AssignMovingAvg_12r
7batch_normalization_79/AssignMovingAvg_1/ReadVariableOp7batch_normalization_79/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_79/batchnorm/ReadVariableOp/batch_normalization_79/batchnorm/ReadVariableOp2j
3batch_normalization_79/batchnorm/mul/ReadVariableOp3batch_normalization_79/batchnorm/mul/ReadVariableOp2P
&batch_normalization_80/AssignMovingAvg&batch_normalization_80/AssignMovingAvg2n
5batch_normalization_80/AssignMovingAvg/ReadVariableOp5batch_normalization_80/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_80/AssignMovingAvg_1(batch_normalization_80/AssignMovingAvg_12r
7batch_normalization_80/AssignMovingAvg_1/ReadVariableOp7batch_normalization_80/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_80/batchnorm/ReadVariableOp/batch_normalization_80/batchnorm/ReadVariableOp2j
3batch_normalization_80/batchnorm/mul/ReadVariableOp3batch_normalization_80/batchnorm/mul/ReadVariableOp2P
&batch_normalization_81/AssignMovingAvg&batch_normalization_81/AssignMovingAvg2n
5batch_normalization_81/AssignMovingAvg/ReadVariableOp5batch_normalization_81/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_81/AssignMovingAvg_1(batch_normalization_81/AssignMovingAvg_12r
7batch_normalization_81/AssignMovingAvg_1/ReadVariableOp7batch_normalization_81/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_81/batchnorm/ReadVariableOp/batch_normalization_81/batchnorm/ReadVariableOp2j
3batch_normalization_81/batchnorm/mul/ReadVariableOp3batch_normalization_81/batchnorm/mul/ReadVariableOp2P
&batch_normalization_82/AssignMovingAvg&batch_normalization_82/AssignMovingAvg2n
5batch_normalization_82/AssignMovingAvg/ReadVariableOp5batch_normalization_82/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_82/AssignMovingAvg_1(batch_normalization_82/AssignMovingAvg_12r
7batch_normalization_82/AssignMovingAvg_1/ReadVariableOp7batch_normalization_82/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_82/batchnorm/ReadVariableOp/batch_normalization_82/batchnorm/ReadVariableOp2j
3batch_normalization_82/batchnorm/mul/ReadVariableOp3batch_normalization_82/batchnorm/mul/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������(

_user_specified_namex
��
�
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7059140
input_1F
8batch_normalization_78_batchnorm_readvariableop_resource:(J
<batch_normalization_78_batchnorm_mul_readvariableop_resource:(H
:batch_normalization_78_batchnorm_readvariableop_1_resource:(H
:batch_normalization_78_batchnorm_readvariableop_2_resource:(9
'dense_65_matmul_readvariableop_resource:(<F
8batch_normalization_79_batchnorm_readvariableop_resource:<J
<batch_normalization_79_batchnorm_mul_readvariableop_resource:<H
:batch_normalization_79_batchnorm_readvariableop_1_resource:<H
:batch_normalization_79_batchnorm_readvariableop_2_resource:<9
'dense_66_matmul_readvariableop_resource:<<F
8batch_normalization_80_batchnorm_readvariableop_resource:<J
<batch_normalization_80_batchnorm_mul_readvariableop_resource:<H
:batch_normalization_80_batchnorm_readvariableop_1_resource:<H
:batch_normalization_80_batchnorm_readvariableop_2_resource:<9
'dense_67_matmul_readvariableop_resource:<<F
8batch_normalization_81_batchnorm_readvariableop_resource:<J
<batch_normalization_81_batchnorm_mul_readvariableop_resource:<H
:batch_normalization_81_batchnorm_readvariableop_1_resource:<H
:batch_normalization_81_batchnorm_readvariableop_2_resource:<9
'dense_68_matmul_readvariableop_resource:<<F
8batch_normalization_82_batchnorm_readvariableop_resource:<J
<batch_normalization_82_batchnorm_mul_readvariableop_resource:<H
:batch_normalization_82_batchnorm_readvariableop_1_resource:<H
:batch_normalization_82_batchnorm_readvariableop_2_resource:<9
'dense_69_matmul_readvariableop_resource:<(6
(dense_69_biasadd_readvariableop_resource:(
identity��/batch_normalization_78/batchnorm/ReadVariableOp�1batch_normalization_78/batchnorm/ReadVariableOp_1�1batch_normalization_78/batchnorm/ReadVariableOp_2�3batch_normalization_78/batchnorm/mul/ReadVariableOp�/batch_normalization_79/batchnorm/ReadVariableOp�1batch_normalization_79/batchnorm/ReadVariableOp_1�1batch_normalization_79/batchnorm/ReadVariableOp_2�3batch_normalization_79/batchnorm/mul/ReadVariableOp�/batch_normalization_80/batchnorm/ReadVariableOp�1batch_normalization_80/batchnorm/ReadVariableOp_1�1batch_normalization_80/batchnorm/ReadVariableOp_2�3batch_normalization_80/batchnorm/mul/ReadVariableOp�/batch_normalization_81/batchnorm/ReadVariableOp�1batch_normalization_81/batchnorm/ReadVariableOp_1�1batch_normalization_81/batchnorm/ReadVariableOp_2�3batch_normalization_81/batchnorm/mul/ReadVariableOp�/batch_normalization_82/batchnorm/ReadVariableOp�1batch_normalization_82/batchnorm/ReadVariableOp_1�1batch_normalization_82/batchnorm/ReadVariableOp_2�3batch_normalization_82/batchnorm/mul/ReadVariableOp�dense_65/MatMul/ReadVariableOp�dense_66/MatMul/ReadVariableOp�dense_67/MatMul/ReadVariableOp�dense_68/MatMul/ReadVariableOp�dense_69/BiasAdd/ReadVariableOp�dense_69/MatMul/ReadVariableOp�
/batch_normalization_78/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_78_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype021
/batch_normalization_78/batchnorm/ReadVariableOp�
&batch_normalization_78/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_78/batchnorm/add/y�
$batch_normalization_78/batchnorm/addAddV27batch_normalization_78/batchnorm/ReadVariableOp:value:0/batch_normalization_78/batchnorm/add/y:output:0*
T0*
_output_shapes
:(2&
$batch_normalization_78/batchnorm/add�
&batch_normalization_78/batchnorm/RsqrtRsqrt(batch_normalization_78/batchnorm/add:z:0*
T0*
_output_shapes
:(2(
&batch_normalization_78/batchnorm/Rsqrt�
3batch_normalization_78/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_78_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype025
3batch_normalization_78/batchnorm/mul/ReadVariableOp�
$batch_normalization_78/batchnorm/mulMul*batch_normalization_78/batchnorm/Rsqrt:y:0;batch_normalization_78/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2&
$batch_normalization_78/batchnorm/mul�
&batch_normalization_78/batchnorm/mul_1Mulinput_1(batch_normalization_78/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2(
&batch_normalization_78/batchnorm/mul_1�
1batch_normalization_78/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_78_batchnorm_readvariableop_1_resource*
_output_shapes
:(*
dtype023
1batch_normalization_78/batchnorm/ReadVariableOp_1�
&batch_normalization_78/batchnorm/mul_2Mul9batch_normalization_78/batchnorm/ReadVariableOp_1:value:0(batch_normalization_78/batchnorm/mul:z:0*
T0*
_output_shapes
:(2(
&batch_normalization_78/batchnorm/mul_2�
1batch_normalization_78/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_78_batchnorm_readvariableop_2_resource*
_output_shapes
:(*
dtype023
1batch_normalization_78/batchnorm/ReadVariableOp_2�
$batch_normalization_78/batchnorm/subSub9batch_normalization_78/batchnorm/ReadVariableOp_2:value:0*batch_normalization_78/batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2&
$batch_normalization_78/batchnorm/sub�
&batch_normalization_78/batchnorm/add_1AddV2*batch_normalization_78/batchnorm/mul_1:z:0(batch_normalization_78/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2(
&batch_normalization_78/batchnorm/add_1�
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02 
dense_65/MatMul/ReadVariableOp�
dense_65/MatMulMatMul*batch_normalization_78/batchnorm/add_1:z:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_65/MatMul�
/batch_normalization_79/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_79_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_79/batchnorm/ReadVariableOp�
&batch_normalization_79/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_79/batchnorm/add/y�
$batch_normalization_79/batchnorm/addAddV27batch_normalization_79/batchnorm/ReadVariableOp:value:0/batch_normalization_79/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_79/batchnorm/add�
&batch_normalization_79/batchnorm/RsqrtRsqrt(batch_normalization_79/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_79/batchnorm/Rsqrt�
3batch_normalization_79/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_79_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_79/batchnorm/mul/ReadVariableOp�
$batch_normalization_79/batchnorm/mulMul*batch_normalization_79/batchnorm/Rsqrt:y:0;batch_normalization_79/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_79/batchnorm/mul�
&batch_normalization_79/batchnorm/mul_1Muldense_65/MatMul:product:0(batch_normalization_79/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_79/batchnorm/mul_1�
1batch_normalization_79/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_79_batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype023
1batch_normalization_79/batchnorm/ReadVariableOp_1�
&batch_normalization_79/batchnorm/mul_2Mul9batch_normalization_79/batchnorm/ReadVariableOp_1:value:0(batch_normalization_79/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_79/batchnorm/mul_2�
1batch_normalization_79/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_79_batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype023
1batch_normalization_79/batchnorm/ReadVariableOp_2�
$batch_normalization_79/batchnorm/subSub9batch_normalization_79/batchnorm/ReadVariableOp_2:value:0*batch_normalization_79/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_79/batchnorm/sub�
&batch_normalization_79/batchnorm/add_1AddV2*batch_normalization_79/batchnorm/mul_1:z:0(batch_normalization_79/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_79/batchnorm/add_1r
ReluRelu*batch_normalization_79/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu�
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02 
dense_66/MatMul/ReadVariableOp�
dense_66/MatMulMatMulRelu:activations:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_66/MatMul�
/batch_normalization_80/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_80_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_80/batchnorm/ReadVariableOp�
&batch_normalization_80/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_80/batchnorm/add/y�
$batch_normalization_80/batchnorm/addAddV27batch_normalization_80/batchnorm/ReadVariableOp:value:0/batch_normalization_80/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_80/batchnorm/add�
&batch_normalization_80/batchnorm/RsqrtRsqrt(batch_normalization_80/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_80/batchnorm/Rsqrt�
3batch_normalization_80/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_80_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_80/batchnorm/mul/ReadVariableOp�
$batch_normalization_80/batchnorm/mulMul*batch_normalization_80/batchnorm/Rsqrt:y:0;batch_normalization_80/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_80/batchnorm/mul�
&batch_normalization_80/batchnorm/mul_1Muldense_66/MatMul:product:0(batch_normalization_80/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_80/batchnorm/mul_1�
1batch_normalization_80/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_80_batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype023
1batch_normalization_80/batchnorm/ReadVariableOp_1�
&batch_normalization_80/batchnorm/mul_2Mul9batch_normalization_80/batchnorm/ReadVariableOp_1:value:0(batch_normalization_80/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_80/batchnorm/mul_2�
1batch_normalization_80/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_80_batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype023
1batch_normalization_80/batchnorm/ReadVariableOp_2�
$batch_normalization_80/batchnorm/subSub9batch_normalization_80/batchnorm/ReadVariableOp_2:value:0*batch_normalization_80/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_80/batchnorm/sub�
&batch_normalization_80/batchnorm/add_1AddV2*batch_normalization_80/batchnorm/mul_1:z:0(batch_normalization_80/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_80/batchnorm/add_1v
Relu_1Relu*batch_normalization_80/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu_1�
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02 
dense_67/MatMul/ReadVariableOp�
dense_67/MatMulMatMulRelu_1:activations:0&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_67/MatMul�
/batch_normalization_81/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_81_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_81/batchnorm/ReadVariableOp�
&batch_normalization_81/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_81/batchnorm/add/y�
$batch_normalization_81/batchnorm/addAddV27batch_normalization_81/batchnorm/ReadVariableOp:value:0/batch_normalization_81/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_81/batchnorm/add�
&batch_normalization_81/batchnorm/RsqrtRsqrt(batch_normalization_81/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_81/batchnorm/Rsqrt�
3batch_normalization_81/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_81_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_81/batchnorm/mul/ReadVariableOp�
$batch_normalization_81/batchnorm/mulMul*batch_normalization_81/batchnorm/Rsqrt:y:0;batch_normalization_81/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_81/batchnorm/mul�
&batch_normalization_81/batchnorm/mul_1Muldense_67/MatMul:product:0(batch_normalization_81/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_81/batchnorm/mul_1�
1batch_normalization_81/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_81_batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype023
1batch_normalization_81/batchnorm/ReadVariableOp_1�
&batch_normalization_81/batchnorm/mul_2Mul9batch_normalization_81/batchnorm/ReadVariableOp_1:value:0(batch_normalization_81/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_81/batchnorm/mul_2�
1batch_normalization_81/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_81_batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype023
1batch_normalization_81/batchnorm/ReadVariableOp_2�
$batch_normalization_81/batchnorm/subSub9batch_normalization_81/batchnorm/ReadVariableOp_2:value:0*batch_normalization_81/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_81/batchnorm/sub�
&batch_normalization_81/batchnorm/add_1AddV2*batch_normalization_81/batchnorm/mul_1:z:0(batch_normalization_81/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_81/batchnorm/add_1v
Relu_2Relu*batch_normalization_81/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu_2�
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02 
dense_68/MatMul/ReadVariableOp�
dense_68/MatMulMatMulRelu_2:activations:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_68/MatMul�
/batch_normalization_82/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_82_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_82/batchnorm/ReadVariableOp�
&batch_normalization_82/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_82/batchnorm/add/y�
$batch_normalization_82/batchnorm/addAddV27batch_normalization_82/batchnorm/ReadVariableOp:value:0/batch_normalization_82/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_82/batchnorm/add�
&batch_normalization_82/batchnorm/RsqrtRsqrt(batch_normalization_82/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_82/batchnorm/Rsqrt�
3batch_normalization_82/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_82_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_82/batchnorm/mul/ReadVariableOp�
$batch_normalization_82/batchnorm/mulMul*batch_normalization_82/batchnorm/Rsqrt:y:0;batch_normalization_82/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_82/batchnorm/mul�
&batch_normalization_82/batchnorm/mul_1Muldense_68/MatMul:product:0(batch_normalization_82/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_82/batchnorm/mul_1�
1batch_normalization_82/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_82_batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype023
1batch_normalization_82/batchnorm/ReadVariableOp_1�
&batch_normalization_82/batchnorm/mul_2Mul9batch_normalization_82/batchnorm/ReadVariableOp_1:value:0(batch_normalization_82/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_82/batchnorm/mul_2�
1batch_normalization_82/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_82_batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype023
1batch_normalization_82/batchnorm/ReadVariableOp_2�
$batch_normalization_82/batchnorm/subSub9batch_normalization_82/batchnorm/ReadVariableOp_2:value:0*batch_normalization_82/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_82/batchnorm/sub�
&batch_normalization_82/batchnorm/add_1AddV2*batch_normalization_82/batchnorm/mul_1:z:0(batch_normalization_82/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_82/batchnorm/add_1v
Relu_3Relu*batch_normalization_82/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu_3�
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource*
_output_shapes

:<(*
dtype02 
dense_69/MatMul/ReadVariableOp�
dense_69/MatMulMatMulRelu_3:activations:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_69/MatMul�
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02!
dense_69/BiasAdd/ReadVariableOp�
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_69/BiasAddt
IdentityIdentitydense_69/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity�

NoOpNoOp0^batch_normalization_78/batchnorm/ReadVariableOp2^batch_normalization_78/batchnorm/ReadVariableOp_12^batch_normalization_78/batchnorm/ReadVariableOp_24^batch_normalization_78/batchnorm/mul/ReadVariableOp0^batch_normalization_79/batchnorm/ReadVariableOp2^batch_normalization_79/batchnorm/ReadVariableOp_12^batch_normalization_79/batchnorm/ReadVariableOp_24^batch_normalization_79/batchnorm/mul/ReadVariableOp0^batch_normalization_80/batchnorm/ReadVariableOp2^batch_normalization_80/batchnorm/ReadVariableOp_12^batch_normalization_80/batchnorm/ReadVariableOp_24^batch_normalization_80/batchnorm/mul/ReadVariableOp0^batch_normalization_81/batchnorm/ReadVariableOp2^batch_normalization_81/batchnorm/ReadVariableOp_12^batch_normalization_81/batchnorm/ReadVariableOp_24^batch_normalization_81/batchnorm/mul/ReadVariableOp0^batch_normalization_82/batchnorm/ReadVariableOp2^batch_normalization_82/batchnorm/ReadVariableOp_12^batch_normalization_82/batchnorm/ReadVariableOp_24^batch_normalization_82/batchnorm/mul/ReadVariableOp^dense_65/MatMul/ReadVariableOp^dense_66/MatMul/ReadVariableOp^dense_67/MatMul/ReadVariableOp^dense_68/MatMul/ReadVariableOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������(: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_78/batchnorm/ReadVariableOp/batch_normalization_78/batchnorm/ReadVariableOp2f
1batch_normalization_78/batchnorm/ReadVariableOp_11batch_normalization_78/batchnorm/ReadVariableOp_12f
1batch_normalization_78/batchnorm/ReadVariableOp_21batch_normalization_78/batchnorm/ReadVariableOp_22j
3batch_normalization_78/batchnorm/mul/ReadVariableOp3batch_normalization_78/batchnorm/mul/ReadVariableOp2b
/batch_normalization_79/batchnorm/ReadVariableOp/batch_normalization_79/batchnorm/ReadVariableOp2f
1batch_normalization_79/batchnorm/ReadVariableOp_11batch_normalization_79/batchnorm/ReadVariableOp_12f
1batch_normalization_79/batchnorm/ReadVariableOp_21batch_normalization_79/batchnorm/ReadVariableOp_22j
3batch_normalization_79/batchnorm/mul/ReadVariableOp3batch_normalization_79/batchnorm/mul/ReadVariableOp2b
/batch_normalization_80/batchnorm/ReadVariableOp/batch_normalization_80/batchnorm/ReadVariableOp2f
1batch_normalization_80/batchnorm/ReadVariableOp_11batch_normalization_80/batchnorm/ReadVariableOp_12f
1batch_normalization_80/batchnorm/ReadVariableOp_21batch_normalization_80/batchnorm/ReadVariableOp_22j
3batch_normalization_80/batchnorm/mul/ReadVariableOp3batch_normalization_80/batchnorm/mul/ReadVariableOp2b
/batch_normalization_81/batchnorm/ReadVariableOp/batch_normalization_81/batchnorm/ReadVariableOp2f
1batch_normalization_81/batchnorm/ReadVariableOp_11batch_normalization_81/batchnorm/ReadVariableOp_12f
1batch_normalization_81/batchnorm/ReadVariableOp_21batch_normalization_81/batchnorm/ReadVariableOp_22j
3batch_normalization_81/batchnorm/mul/ReadVariableOp3batch_normalization_81/batchnorm/mul/ReadVariableOp2b
/batch_normalization_82/batchnorm/ReadVariableOp/batch_normalization_82/batchnorm/ReadVariableOp2f
1batch_normalization_82/batchnorm/ReadVariableOp_11batch_normalization_82/batchnorm/ReadVariableOp_12f
1batch_normalization_82/batchnorm/ReadVariableOp_21batch_normalization_82/batchnorm/ReadVariableOp_22j
3batch_normalization_82/batchnorm/mul/ReadVariableOp3batch_normalization_82/batchnorm/mul/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������(
!
_user_specified_name	input_1
�
~
*__inference_dense_67_layer_call_fn_7060006

inputs
unknown:<<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_67_layer_call_and_return_conditional_losses_70581532
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������<: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_79_layer_call_and_return_conditional_losses_7057510

inputs5
'assignmovingavg_readvariableop_resource:<7
)assignmovingavg_1_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:</
!batchnorm_readvariableop_resource:<
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:<2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������<2
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

:<*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:<*
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
:<*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2
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
:<*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_81_layer_call_and_return_conditional_losses_7059820

inputs/
!batchnorm_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:<1
#batchnorm_readvariableop_1_resource:<1
#batchnorm_readvariableop_2_resource:<
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_78_layer_call_fn_7059623

inputs
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_78_layer_call_and_return_conditional_losses_70572822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������(: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�

�
E__inference_dense_69_layer_call_and_return_conditional_losses_7060030

inputs0
matmul_readvariableop_resource:<(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
~
*__inference_dense_65_layer_call_fn_7059978

inputs
unknown:(<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_70581112
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������(: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_81_layer_call_and_return_conditional_losses_7057842

inputs5
'assignmovingavg_readvariableop_resource:<7
)assignmovingavg_1_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:</
!batchnorm_readvariableop_resource:<
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:<2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������<2
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

:<*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:<*
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
:<*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2
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
:<*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_81_layer_call_fn_7059869

inputs
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_81_layer_call_and_return_conditional_losses_70577802
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_80_layer_call_and_return_conditional_losses_7057676

inputs5
'assignmovingavg_readvariableop_resource:<7
)assignmovingavg_1_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:</
!batchnorm_readvariableop_resource:<
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:<2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������<2
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

:<*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:<*
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
:<*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2
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
:<*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_79_layer_call_and_return_conditional_losses_7057448

inputs/
!batchnorm_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:<1
#batchnorm_readvariableop_1_resource:<1
#batchnorm_readvariableop_2_resource:<
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
E__inference_dense_65_layer_call_and_return_conditional_losses_7058111

inputs0
matmul_readvariableop_resource:(<
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������(: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
��
�
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7059326
input_1L
>batch_normalization_78_assignmovingavg_readvariableop_resource:(N
@batch_normalization_78_assignmovingavg_1_readvariableop_resource:(J
<batch_normalization_78_batchnorm_mul_readvariableop_resource:(F
8batch_normalization_78_batchnorm_readvariableop_resource:(9
'dense_65_matmul_readvariableop_resource:(<L
>batch_normalization_79_assignmovingavg_readvariableop_resource:<N
@batch_normalization_79_assignmovingavg_1_readvariableop_resource:<J
<batch_normalization_79_batchnorm_mul_readvariableop_resource:<F
8batch_normalization_79_batchnorm_readvariableop_resource:<9
'dense_66_matmul_readvariableop_resource:<<L
>batch_normalization_80_assignmovingavg_readvariableop_resource:<N
@batch_normalization_80_assignmovingavg_1_readvariableop_resource:<J
<batch_normalization_80_batchnorm_mul_readvariableop_resource:<F
8batch_normalization_80_batchnorm_readvariableop_resource:<9
'dense_67_matmul_readvariableop_resource:<<L
>batch_normalization_81_assignmovingavg_readvariableop_resource:<N
@batch_normalization_81_assignmovingavg_1_readvariableop_resource:<J
<batch_normalization_81_batchnorm_mul_readvariableop_resource:<F
8batch_normalization_81_batchnorm_readvariableop_resource:<9
'dense_68_matmul_readvariableop_resource:<<L
>batch_normalization_82_assignmovingavg_readvariableop_resource:<N
@batch_normalization_82_assignmovingavg_1_readvariableop_resource:<J
<batch_normalization_82_batchnorm_mul_readvariableop_resource:<F
8batch_normalization_82_batchnorm_readvariableop_resource:<9
'dense_69_matmul_readvariableop_resource:<(6
(dense_69_biasadd_readvariableop_resource:(
identity��&batch_normalization_78/AssignMovingAvg�5batch_normalization_78/AssignMovingAvg/ReadVariableOp�(batch_normalization_78/AssignMovingAvg_1�7batch_normalization_78/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_78/batchnorm/ReadVariableOp�3batch_normalization_78/batchnorm/mul/ReadVariableOp�&batch_normalization_79/AssignMovingAvg�5batch_normalization_79/AssignMovingAvg/ReadVariableOp�(batch_normalization_79/AssignMovingAvg_1�7batch_normalization_79/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_79/batchnorm/ReadVariableOp�3batch_normalization_79/batchnorm/mul/ReadVariableOp�&batch_normalization_80/AssignMovingAvg�5batch_normalization_80/AssignMovingAvg/ReadVariableOp�(batch_normalization_80/AssignMovingAvg_1�7batch_normalization_80/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_80/batchnorm/ReadVariableOp�3batch_normalization_80/batchnorm/mul/ReadVariableOp�&batch_normalization_81/AssignMovingAvg�5batch_normalization_81/AssignMovingAvg/ReadVariableOp�(batch_normalization_81/AssignMovingAvg_1�7batch_normalization_81/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_81/batchnorm/ReadVariableOp�3batch_normalization_81/batchnorm/mul/ReadVariableOp�&batch_normalization_82/AssignMovingAvg�5batch_normalization_82/AssignMovingAvg/ReadVariableOp�(batch_normalization_82/AssignMovingAvg_1�7batch_normalization_82/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_82/batchnorm/ReadVariableOp�3batch_normalization_82/batchnorm/mul/ReadVariableOp�dense_65/MatMul/ReadVariableOp�dense_66/MatMul/ReadVariableOp�dense_67/MatMul/ReadVariableOp�dense_68/MatMul/ReadVariableOp�dense_69/BiasAdd/ReadVariableOp�dense_69/MatMul/ReadVariableOp�
5batch_normalization_78/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_78/moments/mean/reduction_indices�
#batch_normalization_78/moments/meanMeaninput_1>batch_normalization_78/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:(*
	keep_dims(2%
#batch_normalization_78/moments/mean�
+batch_normalization_78/moments/StopGradientStopGradient,batch_normalization_78/moments/mean:output:0*
T0*
_output_shapes

:(2-
+batch_normalization_78/moments/StopGradient�
0batch_normalization_78/moments/SquaredDifferenceSquaredDifferenceinput_14batch_normalization_78/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������(22
0batch_normalization_78/moments/SquaredDifference�
9batch_normalization_78/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_78/moments/variance/reduction_indices�
'batch_normalization_78/moments/varianceMean4batch_normalization_78/moments/SquaredDifference:z:0Bbatch_normalization_78/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:(*
	keep_dims(2)
'batch_normalization_78/moments/variance�
&batch_normalization_78/moments/SqueezeSqueeze,batch_normalization_78/moments/mean:output:0*
T0*
_output_shapes
:(*
squeeze_dims
 2(
&batch_normalization_78/moments/Squeeze�
(batch_normalization_78/moments/Squeeze_1Squeeze0batch_normalization_78/moments/variance:output:0*
T0*
_output_shapes
:(*
squeeze_dims
 2*
(batch_normalization_78/moments/Squeeze_1�
,batch_normalization_78/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_78/AssignMovingAvg/decay�
+batch_normalization_78/AssignMovingAvg/CastCast5batch_normalization_78/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_78/AssignMovingAvg/Cast�
5batch_normalization_78/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_78_assignmovingavg_readvariableop_resource*
_output_shapes
:(*
dtype027
5batch_normalization_78/AssignMovingAvg/ReadVariableOp�
*batch_normalization_78/AssignMovingAvg/subSub=batch_normalization_78/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_78/moments/Squeeze:output:0*
T0*
_output_shapes
:(2,
*batch_normalization_78/AssignMovingAvg/sub�
*batch_normalization_78/AssignMovingAvg/mulMul.batch_normalization_78/AssignMovingAvg/sub:z:0/batch_normalization_78/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:(2,
*batch_normalization_78/AssignMovingAvg/mul�
&batch_normalization_78/AssignMovingAvgAssignSubVariableOp>batch_normalization_78_assignmovingavg_readvariableop_resource.batch_normalization_78/AssignMovingAvg/mul:z:06^batch_normalization_78/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_78/AssignMovingAvg�
.batch_normalization_78/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_78/AssignMovingAvg_1/decay�
-batch_normalization_78/AssignMovingAvg_1/CastCast7batch_normalization_78/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_78/AssignMovingAvg_1/Cast�
7batch_normalization_78/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_78_assignmovingavg_1_readvariableop_resource*
_output_shapes
:(*
dtype029
7batch_normalization_78/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_78/AssignMovingAvg_1/subSub?batch_normalization_78/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_78/moments/Squeeze_1:output:0*
T0*
_output_shapes
:(2.
,batch_normalization_78/AssignMovingAvg_1/sub�
,batch_normalization_78/AssignMovingAvg_1/mulMul0batch_normalization_78/AssignMovingAvg_1/sub:z:01batch_normalization_78/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:(2.
,batch_normalization_78/AssignMovingAvg_1/mul�
(batch_normalization_78/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_78_assignmovingavg_1_readvariableop_resource0batch_normalization_78/AssignMovingAvg_1/mul:z:08^batch_normalization_78/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_78/AssignMovingAvg_1�
&batch_normalization_78/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_78/batchnorm/add/y�
$batch_normalization_78/batchnorm/addAddV21batch_normalization_78/moments/Squeeze_1:output:0/batch_normalization_78/batchnorm/add/y:output:0*
T0*
_output_shapes
:(2&
$batch_normalization_78/batchnorm/add�
&batch_normalization_78/batchnorm/RsqrtRsqrt(batch_normalization_78/batchnorm/add:z:0*
T0*
_output_shapes
:(2(
&batch_normalization_78/batchnorm/Rsqrt�
3batch_normalization_78/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_78_batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype025
3batch_normalization_78/batchnorm/mul/ReadVariableOp�
$batch_normalization_78/batchnorm/mulMul*batch_normalization_78/batchnorm/Rsqrt:y:0;batch_normalization_78/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2&
$batch_normalization_78/batchnorm/mul�
&batch_normalization_78/batchnorm/mul_1Mulinput_1(batch_normalization_78/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2(
&batch_normalization_78/batchnorm/mul_1�
&batch_normalization_78/batchnorm/mul_2Mul/batch_normalization_78/moments/Squeeze:output:0(batch_normalization_78/batchnorm/mul:z:0*
T0*
_output_shapes
:(2(
&batch_normalization_78/batchnorm/mul_2�
/batch_normalization_78/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_78_batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype021
/batch_normalization_78/batchnorm/ReadVariableOp�
$batch_normalization_78/batchnorm/subSub7batch_normalization_78/batchnorm/ReadVariableOp:value:0*batch_normalization_78/batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2&
$batch_normalization_78/batchnorm/sub�
&batch_normalization_78/batchnorm/add_1AddV2*batch_normalization_78/batchnorm/mul_1:z:0(batch_normalization_78/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2(
&batch_normalization_78/batchnorm/add_1�
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

:(<*
dtype02 
dense_65/MatMul/ReadVariableOp�
dense_65/MatMulMatMul*batch_normalization_78/batchnorm/add_1:z:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_65/MatMul�
5batch_normalization_79/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_79/moments/mean/reduction_indices�
#batch_normalization_79/moments/meanMeandense_65/MatMul:product:0>batch_normalization_79/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2%
#batch_normalization_79/moments/mean�
+batch_normalization_79/moments/StopGradientStopGradient,batch_normalization_79/moments/mean:output:0*
T0*
_output_shapes

:<2-
+batch_normalization_79/moments/StopGradient�
0batch_normalization_79/moments/SquaredDifferenceSquaredDifferencedense_65/MatMul:product:04batch_normalization_79/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������<22
0batch_normalization_79/moments/SquaredDifference�
9batch_normalization_79/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_79/moments/variance/reduction_indices�
'batch_normalization_79/moments/varianceMean4batch_normalization_79/moments/SquaredDifference:z:0Bbatch_normalization_79/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2)
'batch_normalization_79/moments/variance�
&batch_normalization_79/moments/SqueezeSqueeze,batch_normalization_79/moments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2(
&batch_normalization_79/moments/Squeeze�
(batch_normalization_79/moments/Squeeze_1Squeeze0batch_normalization_79/moments/variance:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2*
(batch_normalization_79/moments/Squeeze_1�
,batch_normalization_79/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_79/AssignMovingAvg/decay�
+batch_normalization_79/AssignMovingAvg/CastCast5batch_normalization_79/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_79/AssignMovingAvg/Cast�
5batch_normalization_79/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_79_assignmovingavg_readvariableop_resource*
_output_shapes
:<*
dtype027
5batch_normalization_79/AssignMovingAvg/ReadVariableOp�
*batch_normalization_79/AssignMovingAvg/subSub=batch_normalization_79/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_79/moments/Squeeze:output:0*
T0*
_output_shapes
:<2,
*batch_normalization_79/AssignMovingAvg/sub�
*batch_normalization_79/AssignMovingAvg/mulMul.batch_normalization_79/AssignMovingAvg/sub:z:0/batch_normalization_79/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2,
*batch_normalization_79/AssignMovingAvg/mul�
&batch_normalization_79/AssignMovingAvgAssignSubVariableOp>batch_normalization_79_assignmovingavg_readvariableop_resource.batch_normalization_79/AssignMovingAvg/mul:z:06^batch_normalization_79/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_79/AssignMovingAvg�
.batch_normalization_79/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_79/AssignMovingAvg_1/decay�
-batch_normalization_79/AssignMovingAvg_1/CastCast7batch_normalization_79/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_79/AssignMovingAvg_1/Cast�
7batch_normalization_79/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_79_assignmovingavg_1_readvariableop_resource*
_output_shapes
:<*
dtype029
7batch_normalization_79/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_79/AssignMovingAvg_1/subSub?batch_normalization_79/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_79/moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2.
,batch_normalization_79/AssignMovingAvg_1/sub�
,batch_normalization_79/AssignMovingAvg_1/mulMul0batch_normalization_79/AssignMovingAvg_1/sub:z:01batch_normalization_79/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2.
,batch_normalization_79/AssignMovingAvg_1/mul�
(batch_normalization_79/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_79_assignmovingavg_1_readvariableop_resource0batch_normalization_79/AssignMovingAvg_1/mul:z:08^batch_normalization_79/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_79/AssignMovingAvg_1�
&batch_normalization_79/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_79/batchnorm/add/y�
$batch_normalization_79/batchnorm/addAddV21batch_normalization_79/moments/Squeeze_1:output:0/batch_normalization_79/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_79/batchnorm/add�
&batch_normalization_79/batchnorm/RsqrtRsqrt(batch_normalization_79/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_79/batchnorm/Rsqrt�
3batch_normalization_79/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_79_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_79/batchnorm/mul/ReadVariableOp�
$batch_normalization_79/batchnorm/mulMul*batch_normalization_79/batchnorm/Rsqrt:y:0;batch_normalization_79/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_79/batchnorm/mul�
&batch_normalization_79/batchnorm/mul_1Muldense_65/MatMul:product:0(batch_normalization_79/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_79/batchnorm/mul_1�
&batch_normalization_79/batchnorm/mul_2Mul/batch_normalization_79/moments/Squeeze:output:0(batch_normalization_79/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_79/batchnorm/mul_2�
/batch_normalization_79/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_79_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_79/batchnorm/ReadVariableOp�
$batch_normalization_79/batchnorm/subSub7batch_normalization_79/batchnorm/ReadVariableOp:value:0*batch_normalization_79/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_79/batchnorm/sub�
&batch_normalization_79/batchnorm/add_1AddV2*batch_normalization_79/batchnorm/mul_1:z:0(batch_normalization_79/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_79/batchnorm/add_1r
ReluRelu*batch_normalization_79/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu�
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02 
dense_66/MatMul/ReadVariableOp�
dense_66/MatMulMatMulRelu:activations:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_66/MatMul�
5batch_normalization_80/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_80/moments/mean/reduction_indices�
#batch_normalization_80/moments/meanMeandense_66/MatMul:product:0>batch_normalization_80/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2%
#batch_normalization_80/moments/mean�
+batch_normalization_80/moments/StopGradientStopGradient,batch_normalization_80/moments/mean:output:0*
T0*
_output_shapes

:<2-
+batch_normalization_80/moments/StopGradient�
0batch_normalization_80/moments/SquaredDifferenceSquaredDifferencedense_66/MatMul:product:04batch_normalization_80/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������<22
0batch_normalization_80/moments/SquaredDifference�
9batch_normalization_80/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_80/moments/variance/reduction_indices�
'batch_normalization_80/moments/varianceMean4batch_normalization_80/moments/SquaredDifference:z:0Bbatch_normalization_80/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2)
'batch_normalization_80/moments/variance�
&batch_normalization_80/moments/SqueezeSqueeze,batch_normalization_80/moments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2(
&batch_normalization_80/moments/Squeeze�
(batch_normalization_80/moments/Squeeze_1Squeeze0batch_normalization_80/moments/variance:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2*
(batch_normalization_80/moments/Squeeze_1�
,batch_normalization_80/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_80/AssignMovingAvg/decay�
+batch_normalization_80/AssignMovingAvg/CastCast5batch_normalization_80/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_80/AssignMovingAvg/Cast�
5batch_normalization_80/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_80_assignmovingavg_readvariableop_resource*
_output_shapes
:<*
dtype027
5batch_normalization_80/AssignMovingAvg/ReadVariableOp�
*batch_normalization_80/AssignMovingAvg/subSub=batch_normalization_80/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_80/moments/Squeeze:output:0*
T0*
_output_shapes
:<2,
*batch_normalization_80/AssignMovingAvg/sub�
*batch_normalization_80/AssignMovingAvg/mulMul.batch_normalization_80/AssignMovingAvg/sub:z:0/batch_normalization_80/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2,
*batch_normalization_80/AssignMovingAvg/mul�
&batch_normalization_80/AssignMovingAvgAssignSubVariableOp>batch_normalization_80_assignmovingavg_readvariableop_resource.batch_normalization_80/AssignMovingAvg/mul:z:06^batch_normalization_80/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_80/AssignMovingAvg�
.batch_normalization_80/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_80/AssignMovingAvg_1/decay�
-batch_normalization_80/AssignMovingAvg_1/CastCast7batch_normalization_80/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_80/AssignMovingAvg_1/Cast�
7batch_normalization_80/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_80_assignmovingavg_1_readvariableop_resource*
_output_shapes
:<*
dtype029
7batch_normalization_80/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_80/AssignMovingAvg_1/subSub?batch_normalization_80/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_80/moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2.
,batch_normalization_80/AssignMovingAvg_1/sub�
,batch_normalization_80/AssignMovingAvg_1/mulMul0batch_normalization_80/AssignMovingAvg_1/sub:z:01batch_normalization_80/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2.
,batch_normalization_80/AssignMovingAvg_1/mul�
(batch_normalization_80/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_80_assignmovingavg_1_readvariableop_resource0batch_normalization_80/AssignMovingAvg_1/mul:z:08^batch_normalization_80/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_80/AssignMovingAvg_1�
&batch_normalization_80/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_80/batchnorm/add/y�
$batch_normalization_80/batchnorm/addAddV21batch_normalization_80/moments/Squeeze_1:output:0/batch_normalization_80/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_80/batchnorm/add�
&batch_normalization_80/batchnorm/RsqrtRsqrt(batch_normalization_80/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_80/batchnorm/Rsqrt�
3batch_normalization_80/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_80_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_80/batchnorm/mul/ReadVariableOp�
$batch_normalization_80/batchnorm/mulMul*batch_normalization_80/batchnorm/Rsqrt:y:0;batch_normalization_80/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_80/batchnorm/mul�
&batch_normalization_80/batchnorm/mul_1Muldense_66/MatMul:product:0(batch_normalization_80/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_80/batchnorm/mul_1�
&batch_normalization_80/batchnorm/mul_2Mul/batch_normalization_80/moments/Squeeze:output:0(batch_normalization_80/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_80/batchnorm/mul_2�
/batch_normalization_80/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_80_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_80/batchnorm/ReadVariableOp�
$batch_normalization_80/batchnorm/subSub7batch_normalization_80/batchnorm/ReadVariableOp:value:0*batch_normalization_80/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_80/batchnorm/sub�
&batch_normalization_80/batchnorm/add_1AddV2*batch_normalization_80/batchnorm/mul_1:z:0(batch_normalization_80/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_80/batchnorm/add_1v
Relu_1Relu*batch_normalization_80/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu_1�
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02 
dense_67/MatMul/ReadVariableOp�
dense_67/MatMulMatMulRelu_1:activations:0&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_67/MatMul�
5batch_normalization_81/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_81/moments/mean/reduction_indices�
#batch_normalization_81/moments/meanMeandense_67/MatMul:product:0>batch_normalization_81/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2%
#batch_normalization_81/moments/mean�
+batch_normalization_81/moments/StopGradientStopGradient,batch_normalization_81/moments/mean:output:0*
T0*
_output_shapes

:<2-
+batch_normalization_81/moments/StopGradient�
0batch_normalization_81/moments/SquaredDifferenceSquaredDifferencedense_67/MatMul:product:04batch_normalization_81/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������<22
0batch_normalization_81/moments/SquaredDifference�
9batch_normalization_81/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_81/moments/variance/reduction_indices�
'batch_normalization_81/moments/varianceMean4batch_normalization_81/moments/SquaredDifference:z:0Bbatch_normalization_81/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2)
'batch_normalization_81/moments/variance�
&batch_normalization_81/moments/SqueezeSqueeze,batch_normalization_81/moments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2(
&batch_normalization_81/moments/Squeeze�
(batch_normalization_81/moments/Squeeze_1Squeeze0batch_normalization_81/moments/variance:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2*
(batch_normalization_81/moments/Squeeze_1�
,batch_normalization_81/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_81/AssignMovingAvg/decay�
+batch_normalization_81/AssignMovingAvg/CastCast5batch_normalization_81/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_81/AssignMovingAvg/Cast�
5batch_normalization_81/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_81_assignmovingavg_readvariableop_resource*
_output_shapes
:<*
dtype027
5batch_normalization_81/AssignMovingAvg/ReadVariableOp�
*batch_normalization_81/AssignMovingAvg/subSub=batch_normalization_81/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_81/moments/Squeeze:output:0*
T0*
_output_shapes
:<2,
*batch_normalization_81/AssignMovingAvg/sub�
*batch_normalization_81/AssignMovingAvg/mulMul.batch_normalization_81/AssignMovingAvg/sub:z:0/batch_normalization_81/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2,
*batch_normalization_81/AssignMovingAvg/mul�
&batch_normalization_81/AssignMovingAvgAssignSubVariableOp>batch_normalization_81_assignmovingavg_readvariableop_resource.batch_normalization_81/AssignMovingAvg/mul:z:06^batch_normalization_81/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_81/AssignMovingAvg�
.batch_normalization_81/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_81/AssignMovingAvg_1/decay�
-batch_normalization_81/AssignMovingAvg_1/CastCast7batch_normalization_81/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_81/AssignMovingAvg_1/Cast�
7batch_normalization_81/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_81_assignmovingavg_1_readvariableop_resource*
_output_shapes
:<*
dtype029
7batch_normalization_81/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_81/AssignMovingAvg_1/subSub?batch_normalization_81/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_81/moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2.
,batch_normalization_81/AssignMovingAvg_1/sub�
,batch_normalization_81/AssignMovingAvg_1/mulMul0batch_normalization_81/AssignMovingAvg_1/sub:z:01batch_normalization_81/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2.
,batch_normalization_81/AssignMovingAvg_1/mul�
(batch_normalization_81/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_81_assignmovingavg_1_readvariableop_resource0batch_normalization_81/AssignMovingAvg_1/mul:z:08^batch_normalization_81/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_81/AssignMovingAvg_1�
&batch_normalization_81/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_81/batchnorm/add/y�
$batch_normalization_81/batchnorm/addAddV21batch_normalization_81/moments/Squeeze_1:output:0/batch_normalization_81/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_81/batchnorm/add�
&batch_normalization_81/batchnorm/RsqrtRsqrt(batch_normalization_81/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_81/batchnorm/Rsqrt�
3batch_normalization_81/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_81_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_81/batchnorm/mul/ReadVariableOp�
$batch_normalization_81/batchnorm/mulMul*batch_normalization_81/batchnorm/Rsqrt:y:0;batch_normalization_81/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_81/batchnorm/mul�
&batch_normalization_81/batchnorm/mul_1Muldense_67/MatMul:product:0(batch_normalization_81/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_81/batchnorm/mul_1�
&batch_normalization_81/batchnorm/mul_2Mul/batch_normalization_81/moments/Squeeze:output:0(batch_normalization_81/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_81/batchnorm/mul_2�
/batch_normalization_81/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_81_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_81/batchnorm/ReadVariableOp�
$batch_normalization_81/batchnorm/subSub7batch_normalization_81/batchnorm/ReadVariableOp:value:0*batch_normalization_81/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_81/batchnorm/sub�
&batch_normalization_81/batchnorm/add_1AddV2*batch_normalization_81/batchnorm/mul_1:z:0(batch_normalization_81/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_81/batchnorm/add_1v
Relu_2Relu*batch_normalization_81/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu_2�
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype02 
dense_68/MatMul/ReadVariableOp�
dense_68/MatMulMatMulRelu_2:activations:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
dense_68/MatMul�
5batch_normalization_82/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_82/moments/mean/reduction_indices�
#batch_normalization_82/moments/meanMeandense_68/MatMul:product:0>batch_normalization_82/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2%
#batch_normalization_82/moments/mean�
+batch_normalization_82/moments/StopGradientStopGradient,batch_normalization_82/moments/mean:output:0*
T0*
_output_shapes

:<2-
+batch_normalization_82/moments/StopGradient�
0batch_normalization_82/moments/SquaredDifferenceSquaredDifferencedense_68/MatMul:product:04batch_normalization_82/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������<22
0batch_normalization_82/moments/SquaredDifference�
9batch_normalization_82/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_82/moments/variance/reduction_indices�
'batch_normalization_82/moments/varianceMean4batch_normalization_82/moments/SquaredDifference:z:0Bbatch_normalization_82/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2)
'batch_normalization_82/moments/variance�
&batch_normalization_82/moments/SqueezeSqueeze,batch_normalization_82/moments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2(
&batch_normalization_82/moments/Squeeze�
(batch_normalization_82/moments/Squeeze_1Squeeze0batch_normalization_82/moments/variance:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2*
(batch_normalization_82/moments/Squeeze_1�
,batch_normalization_82/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_82/AssignMovingAvg/decay�
+batch_normalization_82/AssignMovingAvg/CastCast5batch_normalization_82/AssignMovingAvg/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+batch_normalization_82/AssignMovingAvg/Cast�
5batch_normalization_82/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_82_assignmovingavg_readvariableop_resource*
_output_shapes
:<*
dtype027
5batch_normalization_82/AssignMovingAvg/ReadVariableOp�
*batch_normalization_82/AssignMovingAvg/subSub=batch_normalization_82/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_82/moments/Squeeze:output:0*
T0*
_output_shapes
:<2,
*batch_normalization_82/AssignMovingAvg/sub�
*batch_normalization_82/AssignMovingAvg/mulMul.batch_normalization_82/AssignMovingAvg/sub:z:0/batch_normalization_82/AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2,
*batch_normalization_82/AssignMovingAvg/mul�
&batch_normalization_82/AssignMovingAvgAssignSubVariableOp>batch_normalization_82_assignmovingavg_readvariableop_resource.batch_normalization_82/AssignMovingAvg/mul:z:06^batch_normalization_82/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_82/AssignMovingAvg�
.batch_normalization_82/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_82/AssignMovingAvg_1/decay�
-batch_normalization_82/AssignMovingAvg_1/CastCast7batch_normalization_82/AssignMovingAvg_1/decay:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-batch_normalization_82/AssignMovingAvg_1/Cast�
7batch_normalization_82/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_82_assignmovingavg_1_readvariableop_resource*
_output_shapes
:<*
dtype029
7batch_normalization_82/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_82/AssignMovingAvg_1/subSub?batch_normalization_82/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_82/moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2.
,batch_normalization_82/AssignMovingAvg_1/sub�
,batch_normalization_82/AssignMovingAvg_1/mulMul0batch_normalization_82/AssignMovingAvg_1/sub:z:01batch_normalization_82/AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2.
,batch_normalization_82/AssignMovingAvg_1/mul�
(batch_normalization_82/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_82_assignmovingavg_1_readvariableop_resource0batch_normalization_82/AssignMovingAvg_1/mul:z:08^batch_normalization_82/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_82/AssignMovingAvg_1�
&batch_normalization_82/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2(
&batch_normalization_82/batchnorm/add/y�
$batch_normalization_82/batchnorm/addAddV21batch_normalization_82/moments/Squeeze_1:output:0/batch_normalization_82/batchnorm/add/y:output:0*
T0*
_output_shapes
:<2&
$batch_normalization_82/batchnorm/add�
&batch_normalization_82/batchnorm/RsqrtRsqrt(batch_normalization_82/batchnorm/add:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_82/batchnorm/Rsqrt�
3batch_normalization_82/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_82_batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype025
3batch_normalization_82/batchnorm/mul/ReadVariableOp�
$batch_normalization_82/batchnorm/mulMul*batch_normalization_82/batchnorm/Rsqrt:y:0;batch_normalization_82/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2&
$batch_normalization_82/batchnorm/mul�
&batch_normalization_82/batchnorm/mul_1Muldense_68/MatMul:product:0(batch_normalization_82/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_82/batchnorm/mul_1�
&batch_normalization_82/batchnorm/mul_2Mul/batch_normalization_82/moments/Squeeze:output:0(batch_normalization_82/batchnorm/mul:z:0*
T0*
_output_shapes
:<2(
&batch_normalization_82/batchnorm/mul_2�
/batch_normalization_82/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_82_batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype021
/batch_normalization_82/batchnorm/ReadVariableOp�
$batch_normalization_82/batchnorm/subSub7batch_normalization_82/batchnorm/ReadVariableOp:value:0*batch_normalization_82/batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2&
$batch_normalization_82/batchnorm/sub�
&batch_normalization_82/batchnorm/add_1AddV2*batch_normalization_82/batchnorm/mul_1:z:0(batch_normalization_82/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2(
&batch_normalization_82/batchnorm/add_1v
Relu_3Relu*batch_normalization_82/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������<2
Relu_3�
dense_69/MatMul/ReadVariableOpReadVariableOp'dense_69_matmul_readvariableop_resource*
_output_shapes

:<(*
dtype02 
dense_69/MatMul/ReadVariableOp�
dense_69/MatMulMatMulRelu_3:activations:0&dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_69/MatMul�
dense_69/BiasAdd/ReadVariableOpReadVariableOp(dense_69_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02!
dense_69/BiasAdd/ReadVariableOp�
dense_69/BiasAddBiasAdddense_69/MatMul:product:0'dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
dense_69/BiasAddt
IdentityIdentitydense_69/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity�
NoOpNoOp'^batch_normalization_78/AssignMovingAvg6^batch_normalization_78/AssignMovingAvg/ReadVariableOp)^batch_normalization_78/AssignMovingAvg_18^batch_normalization_78/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_78/batchnorm/ReadVariableOp4^batch_normalization_78/batchnorm/mul/ReadVariableOp'^batch_normalization_79/AssignMovingAvg6^batch_normalization_79/AssignMovingAvg/ReadVariableOp)^batch_normalization_79/AssignMovingAvg_18^batch_normalization_79/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_79/batchnorm/ReadVariableOp4^batch_normalization_79/batchnorm/mul/ReadVariableOp'^batch_normalization_80/AssignMovingAvg6^batch_normalization_80/AssignMovingAvg/ReadVariableOp)^batch_normalization_80/AssignMovingAvg_18^batch_normalization_80/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_80/batchnorm/ReadVariableOp4^batch_normalization_80/batchnorm/mul/ReadVariableOp'^batch_normalization_81/AssignMovingAvg6^batch_normalization_81/AssignMovingAvg/ReadVariableOp)^batch_normalization_81/AssignMovingAvg_18^batch_normalization_81/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_81/batchnorm/ReadVariableOp4^batch_normalization_81/batchnorm/mul/ReadVariableOp'^batch_normalization_82/AssignMovingAvg6^batch_normalization_82/AssignMovingAvg/ReadVariableOp)^batch_normalization_82/AssignMovingAvg_18^batch_normalization_82/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_82/batchnorm/ReadVariableOp4^batch_normalization_82/batchnorm/mul/ReadVariableOp^dense_65/MatMul/ReadVariableOp^dense_66/MatMul/ReadVariableOp^dense_67/MatMul/ReadVariableOp^dense_68/MatMul/ReadVariableOp ^dense_69/BiasAdd/ReadVariableOp^dense_69/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������(: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_78/AssignMovingAvg&batch_normalization_78/AssignMovingAvg2n
5batch_normalization_78/AssignMovingAvg/ReadVariableOp5batch_normalization_78/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_78/AssignMovingAvg_1(batch_normalization_78/AssignMovingAvg_12r
7batch_normalization_78/AssignMovingAvg_1/ReadVariableOp7batch_normalization_78/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_78/batchnorm/ReadVariableOp/batch_normalization_78/batchnorm/ReadVariableOp2j
3batch_normalization_78/batchnorm/mul/ReadVariableOp3batch_normalization_78/batchnorm/mul/ReadVariableOp2P
&batch_normalization_79/AssignMovingAvg&batch_normalization_79/AssignMovingAvg2n
5batch_normalization_79/AssignMovingAvg/ReadVariableOp5batch_normalization_79/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_79/AssignMovingAvg_1(batch_normalization_79/AssignMovingAvg_12r
7batch_normalization_79/AssignMovingAvg_1/ReadVariableOp7batch_normalization_79/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_79/batchnorm/ReadVariableOp/batch_normalization_79/batchnorm/ReadVariableOp2j
3batch_normalization_79/batchnorm/mul/ReadVariableOp3batch_normalization_79/batchnorm/mul/ReadVariableOp2P
&batch_normalization_80/AssignMovingAvg&batch_normalization_80/AssignMovingAvg2n
5batch_normalization_80/AssignMovingAvg/ReadVariableOp5batch_normalization_80/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_80/AssignMovingAvg_1(batch_normalization_80/AssignMovingAvg_12r
7batch_normalization_80/AssignMovingAvg_1/ReadVariableOp7batch_normalization_80/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_80/batchnorm/ReadVariableOp/batch_normalization_80/batchnorm/ReadVariableOp2j
3batch_normalization_80/batchnorm/mul/ReadVariableOp3batch_normalization_80/batchnorm/mul/ReadVariableOp2P
&batch_normalization_81/AssignMovingAvg&batch_normalization_81/AssignMovingAvg2n
5batch_normalization_81/AssignMovingAvg/ReadVariableOp5batch_normalization_81/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_81/AssignMovingAvg_1(batch_normalization_81/AssignMovingAvg_12r
7batch_normalization_81/AssignMovingAvg_1/ReadVariableOp7batch_normalization_81/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_81/batchnorm/ReadVariableOp/batch_normalization_81/batchnorm/ReadVariableOp2j
3batch_normalization_81/batchnorm/mul/ReadVariableOp3batch_normalization_81/batchnorm/mul/ReadVariableOp2P
&batch_normalization_82/AssignMovingAvg&batch_normalization_82/AssignMovingAvg2n
5batch_normalization_82/AssignMovingAvg/ReadVariableOp5batch_normalization_82/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_82/AssignMovingAvg_1(batch_normalization_82/AssignMovingAvg_12r
7batch_normalization_82/AssignMovingAvg_1/ReadVariableOp7batch_normalization_82/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_82/batchnorm/ReadVariableOp/batch_normalization_82/batchnorm/ReadVariableOp2j
3batch_normalization_82/batchnorm/mul/ReadVariableOp3batch_normalization_82/batchnorm/mul/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp2B
dense_69/BiasAdd/ReadVariableOpdense_69/BiasAdd/ReadVariableOp2@
dense_69/MatMul/ReadVariableOpdense_69/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������(
!
_user_specified_name	input_1
�

�
E__inference_dense_69_layer_call_and_return_conditional_losses_7058198

inputs0
matmul_readvariableop_resource:<(-
biasadd_readvariableop_resource:(
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������(2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_79_layer_call_and_return_conditional_losses_7059656

inputs/
!batchnorm_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:<1
#batchnorm_readvariableop_1_resource:<1
#batchnorm_readvariableop_2_resource:<
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_79_layer_call_and_return_conditional_losses_7059692

inputs5
'assignmovingavg_readvariableop_resource:<7
)assignmovingavg_1_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:</
!batchnorm_readvariableop_resource:<
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:<2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������<2
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

:<*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:<*
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
:<*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2
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
:<*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
E__inference_dense_67_layer_call_and_return_conditional_losses_7059999

inputs0
matmul_readvariableop_resource:<<
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMulk
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������<: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_80_layer_call_and_return_conditional_losses_7059774

inputs5
'assignmovingavg_readvariableop_resource:<7
)assignmovingavg_1_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:</
!batchnorm_readvariableop_resource:<
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:<2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������<2
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

:<*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:<*
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
:<*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2
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
:<*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_78_layer_call_and_return_conditional_losses_7057282

inputs/
!batchnorm_readvariableop_resource:(3
%batchnorm_mul_readvariableop_resource:(1
#batchnorm_readvariableop_1_resource:(1
#batchnorm_readvariableop_2_resource:(
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:(2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:(2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:(2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������(: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_78_layer_call_fn_7059636

inputs
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_78_layer_call_and_return_conditional_losses_70573442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������(: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_79_layer_call_fn_7059705

inputs
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_79_layer_call_and_return_conditional_losses_70574482
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_7058742
input_1
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(<
	unknown_4:<
	unknown_5:<
	unknown_6:<
	unknown_7:<
	unknown_8:<<
	unknown_9:<

unknown_10:<

unknown_11:<

unknown_12:<

unknown_13:<<

unknown_14:<

unknown_15:<

unknown_16:<

unknown_17:<

unknown_18:<<

unknown_19:<

unknown_20:<

unknown_21:<

unknown_22:<

unknown_23:<(

unknown_24:(
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
:���������(*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_70572582
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������(: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������(
!
_user_specified_name	input_1
�
�
S__inference_batch_normalization_78_layer_call_and_return_conditional_losses_7059574

inputs/
!batchnorm_readvariableop_resource:(3
%batchnorm_mul_readvariableop_resource:(1
#batchnorm_readvariableop_1_resource:(1
#batchnorm_readvariableop_2_resource:(
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:(2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:(2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:(2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������(: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
9__inference_feed_forward_sub_net_13_layer_call_fn_7059440
x
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(<
	unknown_4:<
	unknown_5:<
	unknown_6:<
	unknown_7:<
	unknown_8:<<
	unknown_9:<

unknown_10:<

unknown_11:<

unknown_12:<

unknown_13:<<

unknown_14:<

unknown_15:<

unknown_16:<

unknown_17:<

unknown_18:<<

unknown_19:<

unknown_20:<

unknown_21:<

unknown_22:<

unknown_23:<(

unknown_24:(
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
:���������(*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_70582052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������(2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������(: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������(

_user_specified_namex
�,
�
S__inference_batch_normalization_78_layer_call_and_return_conditional_losses_7057344

inputs5
'assignmovingavg_readvariableop_resource:(7
)assignmovingavg_1_readvariableop_resource:(3
%batchnorm_mul_readvariableop_resource:(/
!batchnorm_readvariableop_resource:(
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:(*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:(2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������(2
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

:(*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:(*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:(*
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
:(*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:(2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:(2
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
:(*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:(2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:(2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:(2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:(2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:(2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:(2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:(*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:(2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������(2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������(2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������(: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������(
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_79_layer_call_fn_7059718

inputs
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_79_layer_call_and_return_conditional_losses_70575102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_82_layer_call_fn_7059964

inputs
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_82_layer_call_and_return_conditional_losses_70580082
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_82_layer_call_and_return_conditional_losses_7058008

inputs5
'assignmovingavg_readvariableop_resource:<7
)assignmovingavg_1_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:</
!batchnorm_readvariableop_resource:<
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:<2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������<2
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

:<*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:<*
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
:<*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2
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
:<*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�M
�
 __inference__traced_save_7060140
file_prefixe
asavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_beta_read_readvariableope
asavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_gamma_read_readvariableopd
`savev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_beta_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_moving_variance_read_readvariableopk
gsavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_moving_mean_read_readvariableopo
ksavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_moving_variance_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_13_dense_65_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_13_dense_66_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_13_dense_67_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_13_dense_68_kernel_read_readvariableopX
Tsavev2_nonshared_model_1_feed_forward_sub_net_13_dense_69_kernel_read_readvariableopV
Rsavev2_nonshared_model_1_feed_forward_sub_net_13_dense_69_bias_read_readvariableop
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
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0asavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_beta_read_readvariableopasavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_gamma_read_readvariableop`savev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_beta_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_moving_variance_read_readvariableopgsavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_moving_mean_read_readvariableopksavev2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_moving_variance_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_13_dense_65_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_13_dense_66_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_13_dense_67_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_13_dense_68_kernel_read_readvariableopTsavev2_nonshared_model_1_feed_forward_sub_net_13_dense_69_kernel_read_readvariableopRsavev2_nonshared_model_1_feed_forward_sub_net_13_dense_69_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
�: :(:(:<:<:<:<:<:<:<:<:(:(:<:<:<:<:<:<:<:<:(<:<<:<<:<<:<(:(: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:<: 

_output_shapes
:<: 

_output_shapes
:<: 

_output_shapes
:<: 

_output_shapes
:<: 

_output_shapes
:<: 	

_output_shapes
:<: 


_output_shapes
:<: 

_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:<: 

_output_shapes
:<: 

_output_shapes
:<: 

_output_shapes
:<: 

_output_shapes
:<: 

_output_shapes
:<: 

_output_shapes
:<: 

_output_shapes
:<:$ 

_output_shapes

:(<:$ 

_output_shapes

:<<:$ 

_output_shapes

:<<:$ 

_output_shapes

:<<:$ 

_output_shapes

:<(: 

_output_shapes
:(:

_output_shapes
: 
�,
�
S__inference_batch_normalization_82_layer_call_and_return_conditional_losses_7059938

inputs5
'assignmovingavg_readvariableop_resource:<7
)assignmovingavg_1_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:</
!batchnorm_readvariableop_resource:<
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:<2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������<2
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

:<*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:<*
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
:<*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2
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
:<*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_82_layer_call_and_return_conditional_losses_7057946

inputs/
!batchnorm_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:<1
#batchnorm_readvariableop_1_resource:<1
#batchnorm_readvariableop_2_resource:<
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�,
�
S__inference_batch_normalization_81_layer_call_and_return_conditional_losses_7059856

inputs5
'assignmovingavg_readvariableop_resource:<7
)assignmovingavg_1_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:</
!batchnorm_readvariableop_resource:<
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:<*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:<2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������<2
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

:<*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:<*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:<*
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
:<*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/Cast:y:0*
T0*
_output_shapes
:<2
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
:<*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0AssignMovingAvg_1/Cast:y:0*
T0*
_output_shapes
:<2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1k
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_7060228
file_prefixe
Wassignvariableop_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_gamma:(f
Xassignvariableop_1_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_beta:(g
Yassignvariableop_2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_gamma:<f
Xassignvariableop_3_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_beta:<g
Yassignvariableop_4_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_gamma:<f
Xassignvariableop_5_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_beta:<g
Yassignvariableop_6_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_gamma:<f
Xassignvariableop_7_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_beta:<g
Yassignvariableop_8_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_gamma:<f
Xassignvariableop_9_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_beta:<n
`assignvariableop_10_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_moving_mean:(r
dassignvariableop_11_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_moving_variance:(n
`assignvariableop_12_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_moving_mean:<r
dassignvariableop_13_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_moving_variance:<n
`assignvariableop_14_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_moving_mean:<r
dassignvariableop_15_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_moving_variance:<n
`assignvariableop_16_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_moving_mean:<r
dassignvariableop_17_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_moving_variance:<n
`assignvariableop_18_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_moving_mean:<r
dassignvariableop_19_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_moving_variance:<_
Massignvariableop_20_nonshared_model_1_feed_forward_sub_net_13_dense_65_kernel:(<_
Massignvariableop_21_nonshared_model_1_feed_forward_sub_net_13_dense_66_kernel:<<_
Massignvariableop_22_nonshared_model_1_feed_forward_sub_net_13_dense_67_kernel:<<_
Massignvariableop_23_nonshared_model_1_feed_forward_sub_net_13_dense_68_kernel:<<_
Massignvariableop_24_nonshared_model_1_feed_forward_sub_net_13_dense_69_kernel:<(Y
Kassignvariableop_25_nonshared_model_1_feed_forward_sub_net_13_dense_69_bias:(
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
AssignVariableOpAssignVariableOpWassignvariableop_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpXassignvariableop_1_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpYassignvariableop_2_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpXassignvariableop_3_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpYassignvariableop_4_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpXassignvariableop_5_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpYassignvariableop_6_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpXassignvariableop_7_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpYassignvariableop_8_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpXassignvariableop_9_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp`assignvariableop_10_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpdassignvariableop_11_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_78_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp`assignvariableop_12_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpdassignvariableop_13_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_79_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp`assignvariableop_14_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpdassignvariableop_15_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_80_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp`assignvariableop_16_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpdassignvariableop_17_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_81_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp`assignvariableop_18_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpdassignvariableop_19_nonshared_model_1_feed_forward_sub_net_13_batch_normalization_82_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpMassignvariableop_20_nonshared_model_1_feed_forward_sub_net_13_dense_65_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpMassignvariableop_21_nonshared_model_1_feed_forward_sub_net_13_dense_66_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpMassignvariableop_22_nonshared_model_1_feed_forward_sub_net_13_dense_67_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpMassignvariableop_23_nonshared_model_1_feed_forward_sub_net_13_dense_68_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpMassignvariableop_24_nonshared_model_1_feed_forward_sub_net_13_dense_69_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpKassignvariableop_25_nonshared_model_1_feed_forward_sub_net_13_dense_69_biasIdentity_25:output:0"/device:CPU:0*
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
8__inference_batch_normalization_81_layer_call_fn_7059882

inputs
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_81_layer_call_and_return_conditional_losses_70578422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
~
*__inference_dense_66_layer_call_fn_7059992

inputs
unknown:<<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_66_layer_call_and_return_conditional_losses_70581322
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������<: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_82_layer_call_fn_7059951

inputs
unknown:<
	unknown_0:<
	unknown_1:<
	unknown_2:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_82_layer_call_and_return_conditional_losses_70579462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_82_layer_call_and_return_conditional_losses_7059902

inputs/
!batchnorm_readvariableop_resource:<3
%batchnorm_mul_readvariableop_resource:<1
#batchnorm_readvariableop_1_resource:<1
#batchnorm_readvariableop_2_resource:<
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOpk
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB 2�����ư>2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:<2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:<2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:<*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:<2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:<2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:<*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:<2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������<2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������<: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
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
serving_default_input_1:0���������(<
output_10
StatefulPartitionedCall:0���������(tensorflow/serving/predict:��
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
.layer_metrics
	variables
/layer_regularization_losses
trainable_variables

0layers
1non_trainable_variables
regularization_losses
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
T:R(2Fnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/gamma
S:Q(2Enonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/beta
T:R<2Fnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/gamma
S:Q<2Enonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/beta
T:R<2Fnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/gamma
S:Q<2Enonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/beta
T:R<2Fnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/gamma
S:Q<2Enonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/beta
T:R<2Fnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/gamma
S:Q<2Enonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/beta
\:Z( (2Lnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_mean
`:^( (2Pnonshared_model_1/feed_forward_sub_net_13/batch_normalization_78/moving_variance
\:Z< (2Lnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_mean
`:^< (2Pnonshared_model_1/feed_forward_sub_net_13/batch_normalization_79/moving_variance
\:Z< (2Lnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_mean
`:^< (2Pnonshared_model_1/feed_forward_sub_net_13/batch_normalization_80/moving_variance
\:Z< (2Lnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_mean
`:^< (2Pnonshared_model_1/feed_forward_sub_net_13/batch_normalization_81/moving_variance
\:Z< (2Lnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_mean
`:^< (2Pnonshared_model_1/feed_forward_sub_net_13/batch_normalization_82/moving_variance
K:I(<29nonshared_model_1/feed_forward_sub_net_13/dense_65/kernel
K:I<<29nonshared_model_1/feed_forward_sub_net_13/dense_66/kernel
K:I<<29nonshared_model_1/feed_forward_sub_net_13/dense_67/kernel
K:I<<29nonshared_model_1/feed_forward_sub_net_13/dense_68/kernel
K:I<(29nonshared_model_1/feed_forward_sub_net_13/dense_69/kernel
E:C(27nonshared_model_1/feed_forward_sub_net_13/dense_69/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
`layer_metrics
ametrics
3	variables
blayer_regularization_losses
4trainable_variables

clayers
dnon_trainable_variables
5regularization_losses
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
elayer_metrics
fmetrics
8	variables
glayer_regularization_losses
9trainable_variables

hlayers
inon_trainable_variables
:regularization_losses
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
jlayer_metrics
kmetrics
=	variables
llayer_regularization_losses
>trainable_variables

mlayers
nnon_trainable_variables
?regularization_losses
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
olayer_metrics
pmetrics
B	variables
qlayer_regularization_losses
Ctrainable_variables

rlayers
snon_trainable_variables
Dregularization_losses
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
tlayer_metrics
umetrics
G	variables
vlayer_regularization_losses
Htrainable_variables

wlayers
xnon_trainable_variables
Iregularization_losses
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
ylayer_metrics
zmetrics
L	variables
{layer_regularization_losses
Mtrainable_variables

|layers
}non_trainable_variables
Nregularization_losses
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
~layer_metrics
metrics
P	variables
 �layer_regularization_losses
Qtrainable_variables
�layers
�non_trainable_variables
Rregularization_losses
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
�layer_metrics
�metrics
T	variables
 �layer_regularization_losses
Utrainable_variables
�layers
�non_trainable_variables
Vregularization_losses
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
�layer_metrics
�metrics
X	variables
 �layer_regularization_losses
Ytrainable_variables
�layers
�non_trainable_variables
Zregularization_losses
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
�layer_metrics
�metrics
\	variables
 �layer_regularization_losses
]trainable_variables
�layers
�non_trainable_variables
^regularization_losses
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
"__inference__wrapped_model_7057258input_1"�
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
�2�
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7058848
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7059034
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7059140
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7059326�
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
9__inference_feed_forward_sub_net_13_layer_call_fn_7059383
9__inference_feed_forward_sub_net_13_layer_call_fn_7059440
9__inference_feed_forward_sub_net_13_layer_call_fn_7059497
9__inference_feed_forward_sub_net_13_layer_call_fn_7059554�
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
%__inference_signature_wrapper_7058742input_1"�
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
S__inference_batch_normalization_78_layer_call_and_return_conditional_losses_7059574
S__inference_batch_normalization_78_layer_call_and_return_conditional_losses_7059610�
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
8__inference_batch_normalization_78_layer_call_fn_7059623
8__inference_batch_normalization_78_layer_call_fn_7059636�
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
S__inference_batch_normalization_79_layer_call_and_return_conditional_losses_7059656
S__inference_batch_normalization_79_layer_call_and_return_conditional_losses_7059692�
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
8__inference_batch_normalization_79_layer_call_fn_7059705
8__inference_batch_normalization_79_layer_call_fn_7059718�
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
S__inference_batch_normalization_80_layer_call_and_return_conditional_losses_7059738
S__inference_batch_normalization_80_layer_call_and_return_conditional_losses_7059774�
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
8__inference_batch_normalization_80_layer_call_fn_7059787
8__inference_batch_normalization_80_layer_call_fn_7059800�
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
S__inference_batch_normalization_81_layer_call_and_return_conditional_losses_7059820
S__inference_batch_normalization_81_layer_call_and_return_conditional_losses_7059856�
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
8__inference_batch_normalization_81_layer_call_fn_7059869
8__inference_batch_normalization_81_layer_call_fn_7059882�
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
S__inference_batch_normalization_82_layer_call_and_return_conditional_losses_7059902
S__inference_batch_normalization_82_layer_call_and_return_conditional_losses_7059938�
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
8__inference_batch_normalization_82_layer_call_fn_7059951
8__inference_batch_normalization_82_layer_call_fn_7059964�
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
E__inference_dense_65_layer_call_and_return_conditional_losses_7059971�
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
*__inference_dense_65_layer_call_fn_7059978�
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
E__inference_dense_66_layer_call_and_return_conditional_losses_7059985�
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
*__inference_dense_66_layer_call_fn_7059992�
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
E__inference_dense_67_layer_call_and_return_conditional_losses_7059999�
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
*__inference_dense_67_layer_call_fn_7060006�
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
E__inference_dense_68_layer_call_and_return_conditional_losses_7060013�
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
*__inference_dense_68_layer_call_fn_7060020�
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
E__inference_dense_69_layer_call_and_return_conditional_losses_7060030�
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
*__inference_dense_69_layer_call_fn_7060039�
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
"__inference__wrapped_model_7057258�' ("!)$#*&%+,0�-
&�#
!�
input_1���������(
� "3�0
.
output_1"�
output_1���������(�
S__inference_batch_normalization_78_layer_call_and_return_conditional_losses_7059574b3�0
)�&
 �
inputs���������(
p 
� "%�"
�
0���������(
� �
S__inference_batch_normalization_78_layer_call_and_return_conditional_losses_7059610b3�0
)�&
 �
inputs���������(
p
� "%�"
�
0���������(
� �
8__inference_batch_normalization_78_layer_call_fn_7059623U3�0
)�&
 �
inputs���������(
p 
� "����������(�
8__inference_batch_normalization_78_layer_call_fn_7059636U3�0
)�&
 �
inputs���������(
p
� "����������(�
S__inference_batch_normalization_79_layer_call_and_return_conditional_losses_7059656b 3�0
)�&
 �
inputs���������<
p 
� "%�"
�
0���������<
� �
S__inference_batch_normalization_79_layer_call_and_return_conditional_losses_7059692b 3�0
)�&
 �
inputs���������<
p
� "%�"
�
0���������<
� �
8__inference_batch_normalization_79_layer_call_fn_7059705U 3�0
)�&
 �
inputs���������<
p 
� "����������<�
8__inference_batch_normalization_79_layer_call_fn_7059718U 3�0
)�&
 �
inputs���������<
p
� "����������<�
S__inference_batch_normalization_80_layer_call_and_return_conditional_losses_7059738b"!3�0
)�&
 �
inputs���������<
p 
� "%�"
�
0���������<
� �
S__inference_batch_normalization_80_layer_call_and_return_conditional_losses_7059774b!"3�0
)�&
 �
inputs���������<
p
� "%�"
�
0���������<
� �
8__inference_batch_normalization_80_layer_call_fn_7059787U"!3�0
)�&
 �
inputs���������<
p 
� "����������<�
8__inference_batch_normalization_80_layer_call_fn_7059800U!"3�0
)�&
 �
inputs���������<
p
� "����������<�
S__inference_batch_normalization_81_layer_call_and_return_conditional_losses_7059820b$#3�0
)�&
 �
inputs���������<
p 
� "%�"
�
0���������<
� �
S__inference_batch_normalization_81_layer_call_and_return_conditional_losses_7059856b#$3�0
)�&
 �
inputs���������<
p
� "%�"
�
0���������<
� �
8__inference_batch_normalization_81_layer_call_fn_7059869U$#3�0
)�&
 �
inputs���������<
p 
� "����������<�
8__inference_batch_normalization_81_layer_call_fn_7059882U#$3�0
)�&
 �
inputs���������<
p
� "����������<�
S__inference_batch_normalization_82_layer_call_and_return_conditional_losses_7059902b&%3�0
)�&
 �
inputs���������<
p 
� "%�"
�
0���������<
� �
S__inference_batch_normalization_82_layer_call_and_return_conditional_losses_7059938b%&3�0
)�&
 �
inputs���������<
p
� "%�"
�
0���������<
� �
8__inference_batch_normalization_82_layer_call_fn_7059951U&%3�0
)�&
 �
inputs���������<
p 
� "����������<�
8__inference_batch_normalization_82_layer_call_fn_7059964U%&3�0
)�&
 �
inputs���������<
p
� "����������<�
E__inference_dense_65_layer_call_and_return_conditional_losses_7059971['/�,
%�"
 �
inputs���������(
� "%�"
�
0���������<
� |
*__inference_dense_65_layer_call_fn_7059978N'/�,
%�"
 �
inputs���������(
� "����������<�
E__inference_dense_66_layer_call_and_return_conditional_losses_7059985[(/�,
%�"
 �
inputs���������<
� "%�"
�
0���������<
� |
*__inference_dense_66_layer_call_fn_7059992N(/�,
%�"
 �
inputs���������<
� "����������<�
E__inference_dense_67_layer_call_and_return_conditional_losses_7059999[)/�,
%�"
 �
inputs���������<
� "%�"
�
0���������<
� |
*__inference_dense_67_layer_call_fn_7060006N)/�,
%�"
 �
inputs���������<
� "����������<�
E__inference_dense_68_layer_call_and_return_conditional_losses_7060013[*/�,
%�"
 �
inputs���������<
� "%�"
�
0���������<
� |
*__inference_dense_68_layer_call_fn_7060020N*/�,
%�"
 �
inputs���������<
� "����������<�
E__inference_dense_69_layer_call_and_return_conditional_losses_7060030\+,/�,
%�"
 �
inputs���������<
� "%�"
�
0���������(
� }
*__inference_dense_69_layer_call_fn_7060039O+,/�,
%�"
 �
inputs���������<
� "����������(�
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7058848s' ("!)$#*&%+,.�+
$�!
�
x���������(
p 
� "%�"
�
0���������(
� �
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7059034s' (!")#$*%&+,.�+
$�!
�
x���������(
p
� "%�"
�
0���������(
� �
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7059140y' ("!)$#*&%+,4�1
*�'
!�
input_1���������(
p 
� "%�"
�
0���������(
� �
T__inference_feed_forward_sub_net_13_layer_call_and_return_conditional_losses_7059326y' (!")#$*%&+,4�1
*�'
!�
input_1���������(
p
� "%�"
�
0���������(
� �
9__inference_feed_forward_sub_net_13_layer_call_fn_7059383l' ("!)$#*&%+,4�1
*�'
!�
input_1���������(
p 
� "����������(�
9__inference_feed_forward_sub_net_13_layer_call_fn_7059440f' ("!)$#*&%+,.�+
$�!
�
x���������(
p 
� "����������(�
9__inference_feed_forward_sub_net_13_layer_call_fn_7059497f' (!")#$*%&+,.�+
$�!
�
x���������(
p
� "����������(�
9__inference_feed_forward_sub_net_13_layer_call_fn_7059554l' (!")#$*%&+,4�1
*�'
!�
input_1���������(
p
� "����������(�
%__inference_signature_wrapper_7058742�' ("!)$#*&%+,;�8
� 
1�.
,
input_1!�
input_1���������("3�0
.
output_1"�
output_1���������(